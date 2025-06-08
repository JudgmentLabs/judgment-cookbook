import json


class Memory:
    def __init__(self, initial_memory: dict[str, str] = {}):
        self.storage = dict(initial_memory)

    def add_memory(self, args_dict):
        memories_to_add = {}
        if "memories" in args_dict and isinstance(args_dict["memories"], dict):
            memories_to_add = args_dict["memories"]
        elif isinstance(args_dict, dict):
            memories_to_add = args_dict
        else:
            print(
                f"Error: Invalid arguments structure received in add_memory: {args_dict}")
            return "Error: Invalid arguments structure for add_memory."

        if not isinstance(memories_to_add, dict):
            print(
                f"Error: Could not resolve memories to a dictionary: {memories_to_add}")
            return "Error: 'memories' parameter must resolve to a dictionary."

        added_labels = []
        for label, message in memories_to_add.items():
            if isinstance(label, str) and isinstance(message, str):
                self.storage[label] = message
                added_labels.append(label)
            else:
                print(
                    f"Skipping invalid memory item: label={label}, message={message}")

        if not added_labels:
            return "No valid memories provided or added."
        elif len(added_labels) == 1:
            return f"Successfully added memory with label: {added_labels[0]}"
        else:
            return f"Successfully added memories with labels: {', '.join(added_labels)}"

    def get_memory(self, args_dict):
        label = args_dict.get("label")
        if label is None:
            return "Error: 'label' parameter missing in get_memory call."
        return self.storage.get(label, f"No memory found for label: {label}")

    def get_all_memory(self, args_dict=None):
        return self.storage

    def update_memory(self, args_dict):
        label = args_dict.get("label")
        message = args_dict.get("message")
        if label is None or message is None:
            return "Error: 'label' and 'message' parameters are required for update_memory."
        self.storage[label] = message
        return f"Successfully updated memory with label: {label}"
