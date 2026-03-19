# generation_worker
import asyncio
import aiohttp
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from config import (
    COMFYUI_API_PROMPT,
    COMFYUI_API_HISTORY,
    COMFYUI_API_QUEUE,
    COMFYUI_API_INTERRUPT,
    COMFYUI_API_WEBSOCKET,
    WEBSOCKET_INITIAL_TIMEOUT,
    WEBSOCKET_MESSAGE_TIMEOUT,
    WEBSOCKET_MAX_NO_MESSAGE_RETRIES,
    WEBSOCKET_MAX_WAIT_TIME,
)

logger = logging.getLogger(__name__)


class GenerationWorker:
    """
    Send payload to ComfyUI and await completion using WebSocket
    """
    def __init__(self, worker_id, kwargs):
        self.worker_id = worker_id
        self.preprocess_queue = kwargs["preprocess_queue"]
        self.generation_queue = kwargs["generation_queue"]
        self.postprocess_queue = kwargs["postprocess_queue"]
        self.request_store = kwargs["request_store"]
        self.response_store = kwargs["response_store"]
        
        # Configuration
        self.max_wait_time = WEBSOCKET_MAX_WAIT_TIME
        self.ws_url = COMFYUI_API_WEBSOCKET
        self.client_id = f"worker_{worker_id}_{datetime.now().timestamp()}"

    async def work(self):
        logger.info(f"GenerationWorker {self.worker_id}: waiting for jobs")
        while True:
            # Get a task from the job queue
            request_id = await self.generation_queue.get()
            if request_id is None:
                # None is a signal that there are no more tasks
                break

            # Process the job
            logger.info(f"GenerationWorker {self.worker_id} processing job: {request_id}")
            
            try:
                # Get request and result from stores
                request = await self.request_store.get(request_id)
                result = await self.response_store.get(request_id)
                
                if not request:
                    raise Exception(f"Request {request_id} not found in store")
                if not result:
                    raise Exception(f"Result {request_id} not found in store")

                # Check for cancellation
                if result and getattr(result, 'status', '') == 'cancelled':
                    logger.info(f"PreprocessWorker {self.worker_id} skipping cancelled job: {request_id} - jumping to postprocess")
                    await self.postprocess_queue.put(request_id)
                    self.generation_queue.task_done()
                    continue
                    
                # Submit workflow to ComfyUI
                comfyui_job_id = await self.post_workflow(request)
                logger.info(f"Submitted job {request_id} to ComfyUI as {comfyui_job_id}")
                
                # Update status to show generation started
                result.status = "generating"
                result.message = f"Generation started (ComfyUI job: {comfyui_job_id})"
                await self.response_store.set(request_id, result)

                # Check if job is already complete (cached result)
                is_cached = await self.check_if_cached(comfyui_job_id)
                
                if is_cached:
                    logger.info(f"Job {comfyui_job_id} completed immediately (cached result)")
                    execution_result = {
                        "prompt_id": comfyui_job_id,
                        "nodes_executed": [],
                        "progress_updates": [],
                        "completed": True,
                        "cached": True,
                        "error": None
                    }
                else:
                    # Wait for completion using WebSocket
                    execution_result = await self.wait_for_completion_websocket(
                        comfyui_job_id, 
                        request_id
                    )
                
                # Get the final result from ComfyUI history
                comfyui_response = await self.get_result(comfyui_job_id)
                logger.info(f"Retrieved ComfyUI result for {request_id}")
                logger.debug(f"ComfyUI response structure: {json.dumps(comfyui_response, indent=2)[:500]}...")  # First 500 chars
                
                # Update result with success
                result.status = "generated"
                result.message = "Generation complete. Queued for post-processing."
                result.comfyui_response = comfyui_response
                # Store execution details in the comfyui_response if needed
                if execution_result:
                    # Merge execution details into the response
                    if isinstance(result.comfyui_response, dict):
                        result.comfyui_response["execution_details"] = execution_result
                await self.response_store.set(request_id, result)
                
                # Send for post-processing
                await self.postprocess_queue.put(request_id)
                logger.info(f"GenerationWorker {self.worker_id} completed job: {request_id}")
                
            except Exception as e:
                logger.error(f"GenerationWorker {self.worker_id} failed job {request_id}: {e}")
                
                try:
                    # Update result to show failure
                    result = await self.response_store.get(request_id)
                    if result:
                        result.status = "failed"
                        result.message = f"Generation failed: {str(e)}"
                        await self.response_store.set(request_id, result)
                    
                    # Send job to postprocess for cleanup
                    await self.postprocess_queue.put(request_id)
                    
                except Exception as store_error:
                    logger.error(f"Failed to update result store for {request_id}: {store_error}")
            
            finally:
                # Mark the job as complete
                self.generation_queue.task_done()

        logger.info(f"GenerationWorker {self.worker_id} finished")

    async def post_workflow(self, request) -> str:
        """Submit workflow to ComfyUI API"""
        payload = {
            "prompt": request.input.workflow_json,
            "client_id": self.client_id  # Use our worker's client ID
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                logger.debug(f"Posting workflow to {COMFYUI_API_PROMPT}")
                logger.debug(f"Workflow keys: {list(request.input.workflow_json.keys()) if isinstance(request.input.workflow_json, dict) else 'not a dict'}")
                
                async with session.post(
                    COMFYUI_API_PROMPT, 
                    data=json.dumps(payload),
                    headers=headers
                ) as response:
                    
                    response_text = await response.text()
                    logger.debug(f"ComfyUI API response status: {response.status}")
                    logger.debug(f"ComfyUI API response: {response_text[:500]}...")  # First 500 chars
                    
                    if response.status >= 400:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"ComfyUI API error: {response_text}"
                        )
                    
                    response_data = json.loads(response_text)
                    
                    if "prompt_id" in response_data:
                        return response_data["prompt_id"]
                    elif "node_errors" in response_data:
                        error_details = json.dumps(response_data["node_errors"], indent=2)
                        raise Exception(f"ComfyUI node errors: {error_details}")
                    elif "error" in response_data:
                        raise Exception(f"ComfyUI error: {response_data['error']}")
                    else:
                        raise Exception(f"Unexpected response from ComfyUI: {response_data}")
                        
            except asyncio.TimeoutError:
                raise Exception("Timeout posting workflow to ComfyUI")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error posting to ComfyUI: {e}")
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON response from ComfyUI: {e}")

    async def check_if_cached(self, comfyui_job_id: str) -> bool:
        """Check if job is already complete (cached result)"""
        await asyncio.sleep(0.5)  # Give ComfyUI a moment to process
        
        timeout = aiohttp.ClientTimeout(total=5)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{COMFYUI_API_HISTORY}/{comfyui_job_id}"
                async with session.get(url) as response:
                    if response.status == 200:
                        history_data = await response.json()
                        # If we get non-empty data, the job is complete
                        if history_data and history_data != {}:
                            logger.info(f"Job {comfyui_job_id} found in history (cached)")
                            return True
            return False
        except Exception as e:
            logger.debug(f"Error checking cache status: {e}")
            return False
    
    async def check_if_running(self, comfyui_job_id: str) -> bool:
        """Check if job is still queued or running in ComfyUI via /queue endpoint"""
        timeout = aiohttp.ClientTimeout(total=5)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(COMFYUI_API_QUEUE) as response:
                    if response.status == 200:
                        queue_data = await response.json()
                        # Check running jobs
                        for item in queue_data.get("queue_running", []):
                            if len(item) >= 2 and isinstance(item[1], str) and item[1] == comfyui_job_id:
                                return True
                        # Check pending jobs
                        for item in queue_data.get("queue_pending", []):
                            if len(item) >= 2 and isinstance(item[1], str) and item[1] == comfyui_job_id:
                                return True
            return False
        except Exception as e:
            logger.debug(f"Error checking queue status: {e}")
            return False

    async def wait_for_completion_websocket(self, comfyui_job_id: str, request_id: str) -> Dict[str, Any]:
        """
        Wait for ComfyUI job completion using WebSocket connection.
        Reconnects automatically if the WebSocket closes while the job is still running.
        """
        execution_result = {
            "prompt_id": comfyui_job_id,
            "nodes_executed": [],
            "progress_updates": [],
            "completed": False,
            "error": None
        }

        start_time = asyncio.get_event_loop().time()
        max_reconnects = 10
        reconnect_count = 0

        while reconnect_count <= max_reconnects:
            should_reconnect = False

            try:
                completed = await self._ws_listen_loop(
                    comfyui_job_id, request_id, execution_result, start_time
                )
                if completed:
                    return execution_result

                # _ws_listen_loop returned False — WS closed, check if job still running
                try:
                    if await self.check_if_cached(comfyui_job_id):
                        logger.info(f"Job {comfyui_job_id} completed (post-close check)")
                        execution_result["completed"] = True
                        return execution_result
                except Exception:
                    pass

                still_running = False
                try:
                    still_running = await self.check_if_running(comfyui_job_id)
                except Exception:
                    pass

                if still_running:
                    reconnect_count += 1
                    logger.info(f"Job {comfyui_job_id} still running, reconnecting WebSocket "
                              f"({reconnect_count}/{max_reconnects})")
                    await asyncio.sleep(2)
                    should_reconnect = True
                else:
                    # Final history check before giving up
                    await asyncio.sleep(1)
                    try:
                        if await self.check_if_cached(comfyui_job_id):
                            logger.info(f"Job {comfyui_job_id} completed (final check)")
                            execution_result["completed"] = True
                            return execution_result
                    except Exception:
                        pass
                    raise Exception(f"WebSocket closed and job {comfyui_job_id} not found in queue or history")

            except asyncio.TimeoutError:
                logger.warning(f"WebSocket overall timeout for job {comfyui_job_id}")
                await self.cancel_comfyui_job(comfyui_job_id)
                raise Exception(f"WebSocket timeout for job {comfyui_job_id}")
            except aiohttp.ClientError as e:
                # Connection error — check if job finished meanwhile
                try:
                    if await self.check_if_cached(comfyui_job_id):
                        logger.info(f"Job {comfyui_job_id} completed despite connection error")
                        execution_result["completed"] = True
                        return execution_result
                except Exception:
                    pass

                still_running = False
                try:
                    still_running = await self.check_if_running(comfyui_job_id)
                except Exception:
                    pass

                if still_running:
                    reconnect_count += 1
                    logger.info(f"Connection error but job still running, reconnecting "
                              f"({reconnect_count}/{max_reconnects})")
                    await asyncio.sleep(2)
                    should_reconnect = True
                else:
                    await self.cancel_comfyui_job(comfyui_job_id)
                    raise Exception(f"WebSocket connection error: {e}")

            if not should_reconnect:
                break

        if not execution_result["completed"]:
            raise Exception(f"Max WebSocket reconnects ({max_reconnects}) reached for job {comfyui_job_id}")

        return execution_result

    async def _ws_listen_loop(
        self, comfyui_job_id: str, request_id: str,
        execution_result: Dict[str, Any], start_time: float
    ) -> bool:
        """
        Single WebSocket connection listen loop.
        Returns True if job completed, False if WS closed (caller should check if reconnect needed).
        Raises on timeout, error, or cancellation.
        """
        timeout = aiohttp.ClientTimeout(total=self.max_wait_time)
        initial_timeout = WEBSOCKET_INITIAL_TIMEOUT
        message_timeout = WEBSOCKET_MESSAGE_TIMEOUT
        max_no_message_retries = WEBSOCKET_MAX_NO_MESSAGE_RETRIES
        no_message_retry_count = 0

        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.info(f"Connecting to ComfyUI WebSocket at {self.ws_url}")

            async with session.ws_connect(
                self.ws_url,
                params={"clientId": self.client_id}
            ) as ws:
                logger.info(f"WebSocket connected for job {comfyui_job_id}")

                last_update_time = asyncio.get_event_loop().time()
                last_message_time = start_time
                last_cancellation_check = start_time

                while True:
                    try:
                        timeout_duration = initial_timeout if last_message_time == start_time else message_timeout

                        msg = await asyncio.wait_for(
                            ws.receive(),
                            timeout=timeout_duration
                        )

                        last_message_time = asyncio.get_event_loop().time()
                        no_message_retry_count = 0

                        current_time = asyncio.get_event_loop().time()
                        if current_time - last_cancellation_check > 5.0:
                            if await self._check_if_cancelled(request_id):
                                logger.info(f"Job {request_id} cancelled during generation")
                                await self.cancel_comfyui_job(comfyui_job_id)
                                raise Exception(f"Job {request_id} was cancelled during generation")
                            last_cancellation_check = current_time

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                message_type = data.get("type")
                                logger.debug(f"WebSocket message type: {message_type}")

                                if data.get("data", {}).get("prompt_id") == comfyui_job_id:

                                    if message_type == "execution_start":
                                        logger.info(f"Execution started for {comfyui_job_id}")
                                        await self._update_progress(request_id, "Execution started...")

                                    elif message_type == "execution_cached":
                                        nodes = data.get("data", {}).get("nodes", [])
                                        logger.info(f"Using cached results for nodes: {nodes}")
                                        execution_result["nodes_executed"].extend(nodes)

                                    elif message_type == "executing":
                                        node = data.get("data", {}).get("node")
                                        if node:
                                            logger.info(f"Executing node: {node}")
                                            execution_result["nodes_executed"].append(node)
                                            await self._update_progress(request_id, f"Processing node: {node}")
                                        elif data.get("data", {}).get("node") is None:
                                            logger.info(f"Execution complete for {comfyui_job_id}")
                                            execution_result["completed"] = True
                                            return True

                                    elif message_type == "progress":
                                        progress_data = data.get("data", {})
                                        value = progress_data.get("value", 0)
                                        max_value = progress_data.get("max", 100)
                                        progress_pct = (value / max_value * 100) if max_value > 0 else 0
                                        progress_msg = f"Progress: {progress_pct:.1f}% ({value}/{max_value})"
                                        logger.info(f"Progress update: {progress_msg}")
                                        execution_result["progress_updates"].append({
                                            "time": asyncio.get_event_loop().time() - start_time,
                                            "value": value,
                                            "max": max_value,
                                            "percentage": progress_pct
                                        })
                                        current_time = asyncio.get_event_loop().time()
                                        if current_time - last_update_time > 2:
                                            await self._update_progress(request_id, progress_msg)
                                            last_update_time = current_time

                                    elif message_type == "execution_error":
                                        error_data = data.get("data", {})
                                        error_msg = f"Execution error: {error_data}"
                                        logger.error(error_msg)
                                        execution_result["error"] = error_data
                                        raise Exception(error_msg)

                                    elif message_type == "executed":
                                        node = data.get("data", {}).get("node")
                                        output = data.get("data", {}).get("output")
                                        logger.info(f"Node {node} executed successfully")
                                        logger.debug(f"Node output: {json.dumps(output, indent=2)[:500]}...")

                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse WebSocket message: {e}")
                                logger.debug(f"Raw message: {msg.data}")

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            raise Exception(f"WebSocket error: {ws.exception()}")

                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            logger.warning("WebSocket closed — returning to reconnect logic")
                            return False  # Signal caller to check job status and reconnect

                    except asyncio.TimeoutError:
                        no_message_retry_count += 1
                        elapsed = asyncio.get_event_loop().time() - start_time
                        source = "initial" if last_message_time == start_time else "ongoing"
                        logger.warning(f"WebSocket timeout ({source}) for {comfyui_job_id} "
                                    f"(attempt {no_message_retry_count}/{max_no_message_retries}) "
                                    f"after {elapsed:.1f}s")

                        try:
                            if await self.check_if_cached(comfyui_job_id):
                                logger.info(f"Job {comfyui_job_id} is complete (found in history)")
                                execution_result["completed"] = True
                                return True
                        except Exception as check_error:
                            logger.warning(f"Error checking job history: {check_error}")

                        still_running = False
                        try:
                            still_running = await self.check_if_running(comfyui_job_id)
                        except Exception as check_error:
                            logger.warning(f"Error checking queue status: {check_error}")

                        if still_running:
                            logger.info(f"Job {comfyui_job_id} still running in ComfyUI — "
                                      f"continuing to wait (elapsed: {elapsed:.1f}s)")
                            no_message_retry_count = 0
                            await asyncio.sleep(10)
                            continue

                        if no_message_retry_count >= max_no_message_retries:
                            logger.error(f"Job {comfyui_job_id} not found in queue or history "
                                    f"after {max_no_message_retries} attempts")
                            raise Exception(f"Job {comfyui_job_id} disappeared from ComfyUI "
                                        f"after {no_message_retry_count} attempts")

                        wait_time = min(5 * (2 ** (no_message_retry_count - 1)), 30)
                        logger.info(f"Job not found, waiting {wait_time}s before retry "
                                  f"{no_message_retry_count + 1}/{max_no_message_retries}")
                        await asyncio.sleep(wait_time)

                    # Check for overall timeout
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > self.max_wait_time:
                        raise Exception(f"Timeout waiting for job {comfyui_job_id} after {elapsed:.1f} seconds")

        # Should not reach here, but just in case
        return False

    async def _update_progress(self, request_id: str, message: str):
        """Helper to update progress in the response store"""
        try:
            result = await self.response_store.get(request_id)
            if result:
                result.message = message
                await self.response_store.set(request_id, result)
        except Exception as e:
            logger.warning(f"Failed to update progress for {request_id}: {e}")

    async def get_result(self, comfyui_job_id: str) -> Optional[dict]:
        """Get the final result from ComfyUI history"""
        timeout = aiohttp.ClientTimeout(total=30)
        
        # Wait a moment for history to be updated
        await asyncio.sleep(0.5)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{COMFYUI_API_HISTORY}/{comfyui_job_id}"
                logger.debug(f"Fetching result from: {url}")
                
                async with session.get(url) as response:
                    response_text = await response.text()
                    logger.debug(f"History API status: {response.status}")
                    
                    if response.status == 200:
                        history_data = json.loads(response_text)
                        
                        # Check if we got actual data
                        if not history_data or history_data == {}:
                            logger.warning(f"Empty history response for job {comfyui_job_id}")
                            # Try the general history endpoint
                            return await self._get_result_from_general_history(comfyui_job_id)
                        
                        logger.info(f"Retrieved ComfyUI history for job {comfyui_job_id}")
                        return history_data
                    else:
                        raise Exception(f"Failed to get result (status {response.status}): {response_text}")
                        
        except asyncio.TimeoutError:
            raise Exception(f"Timeout getting result for job {comfyui_job_id}")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error getting result: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in result: {e}")

    async def _get_result_from_general_history(self, comfyui_job_id: str) -> Optional[dict]:
        """Fallback: Get result from general history endpoint"""
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Try the general history endpoint
                url = COMFYUI_API_HISTORY.rstrip(f"/{comfyui_job_id}")
                logger.debug(f"Trying general history endpoint: {url}")
                
                async with session.get(url) as response:
                    if response.status == 200:
                        all_history = await response.json()
                        
                        # Look for our job in the history
                        if comfyui_job_id in all_history:
                            logger.info(f"Found job {comfyui_job_id} in general history")
                            return {comfyui_job_id: all_history[comfyui_job_id]}
                        else:
                            logger.warning(f"Job {comfyui_job_id} not found in general history")
                            return {}
                    else:
                        return {}
                        
        except Exception as e:
            logger.error(f"Failed to get result from general history: {e}")
            return {}

    async def _check_if_cancelled(self, request_id: str) -> bool:
        """Check if the job has been cancelled"""
        try:
            result = await self.response_store.get(request_id)
            return result and getattr(result, 'status', '') == 'cancelled'
        except Exception as e:
            logger.warning(f"Error checking cancellation status for {request_id}: {e}")
            return False

    async def cancel_comfyui_job(self, comfyui_job_id: str):
        """Cancel a running job in ComfyUI"""
        try:       
            if not COMFYUI_API_INTERRUPT:
                logger.warning("COMFYUI_API_INTERRUPT not configured, cannot cancel job")
                return False
                
            payload = {
                "prompt_id": comfyui_job_id
            }
            
            headers = {
                'Content-Type': 'application/json'
            }
                
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                cancel_url = COMFYUI_API_INTERRUPT
                
                async with session.post(
                    cancel_url,
                    data=json.dumps(payload),
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        logger.info(f"Successfully cancelled ComfyUI job {comfyui_job_id}")
                        return True
                    else:
                        response_text = await response.text()
                        logger.warning(f"Failed to cancel ComfyUI job {comfyui_job_id}: HTTP {response.status} - {response_text}")
                        return False
                    
        except Exception as e:
            logger.error(f"Error cancelling ComfyUI job {comfyui_job_id}: {e}")
            return False
