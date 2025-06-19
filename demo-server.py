#!/usr/bin/env python3
import http.server
import socketserver
import os
import webbrowser

PORT = 3002

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/demo.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        http.server.SimpleHTTPRequestHandler.end_headers(self)

print(f"🚀 Starting PPO Course Demo Server on port {PORT}")
print(f"📍 Visit http://localhost:{PORT} in your browser")
print(f"")
print(f"The demo includes:")
print(f"  1. Neural Network Visualizer - Click neurons to see values")
print(f"  2. PPO Algorithm Stepper - Step through the algorithm")
print(f"  3. Interactive Grid World - Play the game!")
print(f"  4. Progress Tracking - See gamification features")
print(f"  5. Code Playground Preview - Syntax highlighting")
print(f"")
print(f"Press Ctrl+C to stop the server")
print(f"")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    # Try to open browser automatically
    try:
        webbrowser.open(f'http://localhost:{PORT}')
    except:
        pass
    
    httpd.serve_forever()