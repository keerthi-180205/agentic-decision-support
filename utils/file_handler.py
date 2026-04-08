import pandas as pd
import json

def load_file(file):
    try:
        # -------------------------------
        # CSV FILE
        # -------------------------------
        if file.name.endswith(".csv"):
            return pd.read_csv(file)

        # -------------------------------
        # EXCEL FILE
        # -------------------------------
        elif file.name.endswith(".xlsx") or file.name.endswith(".xls"):
            return pd.read_excel(file, engine="openpyxl")

        # -------------------------------
        # JSON FILE (ROBUST HANDLING)
        # -------------------------------
        elif file.name.endswith(".json"):

            try:
                # Try simple JSON (tabular)
                return pd.read_json(file)

            except:
                # Reset pointer
                file.seek(0)

                data = json.load(file)

                # Case 1: List of dictionaries (BEST CASE)
                if isinstance(data, list):
                    return pd.json_normalize(data)

                # Case 2: Dictionary format
                elif isinstance(data, dict):
                    try:
                        return pd.DataFrame(data)
                    except:
                        return pd.json_normalize(data)

                else:
                    return None

        # -------------------------------
        # UNSUPPORTED FILE
        # -------------------------------
        else:
            return None

    except Exception as e:
        print("File loading error:", e)
        return None