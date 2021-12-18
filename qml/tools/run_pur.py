from pur import update_requirements

if __name__ == "__main__":
    messages = [
        x[0]["message"]
        for x in update_requirements(input_file="requirements.txt").values()
    ]
