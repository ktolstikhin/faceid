import cv2


class VideoStream:

    def __init__(self, path='/dev/video0', size=(640, 480)):
        self.path = path
        self.size = size
        self.cap = self.capture_stream()

    def __del__(self):
        cap = getattr(self, 'cap', None)

        if cap is not None:
            cap.release()

    def capture_stream(self):
        cap = cv2.VideoCapture(self.path)

        if self.path.startswith('/dev/'):
            width, height = self.size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    def read(self):
        success, frame = self.cap.read()

        if not success:
            return

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def draw_box(frame, box, color=(0, 255, 0)):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness=2)

    @staticmethod
    def draw_text(frame, text, anchor=None, color=(0, 255, 0)):

        if anchor is None:
            height = frame.shape[0]
            anchor = (5, height - 5)

        cv2.putText(frame, text, anchor, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=color, thickness=2)

    @staticmethod
    def show(frame, title=None, size=None):

        if frame is None:
            return

        if size is not None:
            frame = cv2.resize(frame, size)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, frame)

    @staticmethod
    def wait_key(timeout=1):
        return chr(cv2.waitKey(timeout) & 0xFF)

    @staticmethod
    def close_windows():
        cv2.destroyAllWindows()

