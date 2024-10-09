import socket
import time
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('192.168.85.130', 10000)
while True:
    message = input('Enter message: ')
    server_socket.sendto(bytes([int(message)]), server_address)
    time.sleep(0.5)
server_socket.close()