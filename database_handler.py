import firebase_admin
from firebase_admin import credentials, firestore
import json

class DatabaseHandler:
    def __init__(self, credentials_path):
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    # UPDATED METHOD
    def save_interaction(self, interaction_type, input_data, output_data):
        """
        Save an interaction to the database.
        
        Args:
            interaction_type (str): Type of interaction (e.g., 'text_detection', 'object_detection').
            input_data (str): The input data for the interaction.
            output_data (str): The output or processed data from the interaction.
        """
        doc_ref = self.db.collection('interactions').document()
        doc_ref.set({
            'type': interaction_type,
            'input': input_data,
            'output': output_data,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        print(f"Interaction of type '{interaction_type}' saved to database")

    def cleanup(self):
        """Cleanup resources if needed."""
        # Firebase Admin SDK doesn't require explicit cleanup
        pass