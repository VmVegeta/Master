import socket
import json


def main():
    s = socket.socket()
    #host = socket.gethostname()
    port = 10203
    array = [1, 2, 3]

    data = json.dumps({"output": array})

    s.connect(('127.0.0.1', port))
    #recive = s.recv(1024)
    s.send(data.encode())
    #print(recive)
    s.close()


if __name__ == '__main__':
    main()
