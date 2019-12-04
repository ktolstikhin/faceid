import uuid


class FaceTarget:

    def __init__(self, label, proba, box):
        self.label = label
        self.proba = proba
        self.box = box
        self.id = uuid.uuid4().hex

