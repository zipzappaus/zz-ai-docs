from typing import IO
import os
import shutil
from app.core.interfaces import DocumentStorage

# Note: We need to define DocumentStorage interface in core/interfaces.py first, 
# but for now I'll assume it exists or I'll add it here for this specific file 
# and then refactor interfaces.py if needed. 
# Actually, I missed adding DocumentStorage to interfaces.py in the previous step.
# I will add it to interfaces.py in a separate step or just implicitly here if I was lazy, 
# but I should do it right.
# Let's fix interfaces.py first in the next step. 
# For now, I'll write this file assuming the interface exists.

class LocalDocumentStorage: # Implements DocumentStorage
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    async def upload(self, file: IO, filename: str) -> str:
        file_path = os.path.join(self.upload_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)
        return file_path

    async def get_url(self, file_path: str) -> str:
        # For local storage, just return the path
        return file_path
