import socketio
import asyncio
from typing import Dict, Set
import json

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*"
)

# Create ASGI app
sio_app = socketio.ASGIApp(sio)

# Store active sessions
active_sessions: Dict[str, Set[str]] = {}

@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    print(f"Client connected: {sid}")
    await sio.emit('connected', {'sid': sid}, room=sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    print(f"Client disconnected: {sid}")
    # Remove from all training sessions
    for session_id in list(active_sessions.keys()):
        if sid in active_sessions[session_id]:
            active_sessions[session_id].remove(sid)
            if not active_sessions[session_id]:
                del active_sessions[session_id]

@sio.event
async def join_training_session(sid, data):
    """Join a training session room"""
    session_id = data.get('session_id')
    if not session_id:
        await sio.emit('error', {'message': 'No session_id provided'}, room=sid)
        return
    
    # Add to session
    if session_id not in active_sessions:
        active_sessions[session_id] = set()
    active_sessions[session_id].add(sid)
    
    # Join Socket.IO room
    await sio.enter_room(sid, f"training_{session_id}")
    await sio.emit('joined_session', {'session_id': session_id}, room=sid)
    
    # Start sending mock training updates
    asyncio.create_task(send_training_updates(session_id))

@sio.event
async def leave_training_session(sid, data):
    """Leave a training session room"""
    session_id = data.get('session_id')
    if not session_id:
        return
    
    # Remove from session
    if session_id in active_sessions and sid in active_sessions[session_id]:
        active_sessions[session_id].remove(sid)
        if not active_sessions[session_id]:
            del active_sessions[session_id]
    
    # Leave Socket.IO room
    await sio.leave_room(sid, f"training_{session_id}")
    await sio.emit('left_session', {'session_id': session_id}, room=sid)

async def send_training_updates(session_id: str):
    """Send mock training updates to all clients in a session"""
    step = 0
    max_steps = 100
    
    while session_id in active_sessions and active_sessions[session_id]:
        # Simulate training metrics
        metrics = {
            'step': step,
            'episode_reward': 10 + step * 0.5 + (5 * (step % 10) / 10),
            'loss': max(0.1, 0.5 - step * 0.004),
            'kl_divergence': 0.01 + 0.002 * (step % 5) / 5,
            'entropy': max(0.5, 1.0 - step * 0.005),
            'value_loss': max(0.05, 0.3 - step * 0.0025),
            'policy_loss': max(0.08, 0.4 - step * 0.0035)
        }
        
        # Send to all clients in the session
        await sio.emit(
            'training_update',
            {
                'session_id': session_id,
                'metrics': metrics,
                'timestamp': asyncio.get_event_loop().time()
            },
            room=f"training_{session_id}"
        )
        
        step += 1
        if step >= max_steps:
            # Training complete
            await sio.emit(
                'training_complete',
                {
                    'session_id': session_id,
                    'final_metrics': metrics
                },
                room=f"training_{session_id}"
            )
            break
        
        # Wait before next update
        await asyncio.sleep(0.5)

@sio.event
async def request_network_update(sid, data):
    """Send neural network weights and activations"""
    session_id = data.get('session_id')
    
    # Mock network data
    network_data = {
        'actor_weights': [
            [[0.1, 0.2, -0.3, 0.4], [0.5, -0.6, 0.7, -0.8]],
            [[0.2, -0.4], [0.6, -0.8], [0.3, -0.5], [0.7, -0.9]]
        ],
        'critic_weights': [
            [[0.15, 0.25, -0.35, 0.45], [0.55, -0.65]],
            [[0.3], [-0.7], [0.4], [-0.8], [0.5], [-0.9]]
        ],
        'actor_activations': [
            [0.8, 0.6, 0.7, 0.9],
            [0.7, 0.8]
        ],
        'critic_activations': [
            [0.6, 0.7, 0.8, 0.9, 0.7, 0.8],
            [0.85]
        ]
    }
    
    await sio.emit('network_update', network_data, room=sid)