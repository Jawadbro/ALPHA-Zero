import firebase_admin
from firebase_admin import credentials, firestore

class DatabaseHandler:
    def __init__(self, credentials_path):
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def save_interaction(self, query, response):
        doc_ref = self.db.collection('memories').document()
        doc_ref.set({
            'query': query,
            'response': response,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        print("Interaction saved to Firebase")

    def cleanup(self):
        """Cleanup resources if needed (currently does nothing)."""
        # Currently, there's nothing to clean up in Firebase Admin SDK,
        # but you could add logic here if needed in the future.
        pass
