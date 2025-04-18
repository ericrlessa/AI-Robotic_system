import time
from web.dashboard import Dashboard


def main():
    dash = Dashboard()
    dash.run()

    print("Dashboard started. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        # Optional: call dash.stop() if you implement cleanup logic
        # dash.stop()


if __name__ == "__main__":
    main()
