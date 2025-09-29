from urllib.parse import quote_plus

class Config:
    # URL encode the password to handle special characters
    username = "DarshanaDayare"
    password = quote_plus("Darshana@2025")
    cluster = "cluster0.kj9z9zv.mongodb.net"
    MONGODB_URI = f"mongodb+srv://{username}:{password}@{cluster}/recruitment_portal?retryWrites=true&w=majority&appName=Cluster0"
    SECRET_KEY = 'your-secret-key-here'  # Change this to a secure secret key
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
    # GridFS collection names
    FS_FILES_COLLECTION = 'fs.files'
    FS_CHUNKS_COLLECTION = 'fs.chunks'
