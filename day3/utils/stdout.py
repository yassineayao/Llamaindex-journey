def save_note(note: str):
    try:
        print("Saving notes")
        with open("./notes.txt", "w") as f:
            f.write(note)
        return "Note saved"
    except Exception as e:
        print(e)
        return "Note not saved"
