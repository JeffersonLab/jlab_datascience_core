import socket

# check internet connection: this is useful as OUD RRCE does not allow internet access
def internet_available(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    Check if internet is available by attempting to connect to a known server.
    """
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
        
        print("Internet connection available")
        return True
    
    except socket.timeout:
        print("No Internet: Connection timed out")
        return False
    
    except socket.gaierror:
        print("No Internet: Address-related error connecting to server")
        return False
    
    except socket.error as e:
        print(f"No Internet: Network error: {e}")
        return False
