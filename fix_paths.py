import sqlite3

def ultimate_path_fixer():
    print("Scanning the entire database for hidden paths...")
    conn = sqlite3.connect("mlflow.db")
    cursor = conn.cursor()

    # Database ki saari tables nikalna
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    
    changes = 0
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [c[1] for c in cursor.fetchall()]
        
        # Har table ke har column mein check karna
        for col in columns:
            try:
                cursor.execute(f"SELECT rowid, {col} FROM {table} WHERE {col} LIKE '%mlruns%'")
                for rowid, val in cursor.fetchall():
                    if val and isinstance(val, str):
                        val_clean = val.replace('\\', '/')
                        if '/mlruns/' in val_clean:
                            right_side = val_clean.split('/mlruns/')[1]
                            new_val = f"file:///app/mlruns/{right_side}"
                            
                            # Agar purana path hai toh hi update karo
                            if new_val != val:
                                cursor.execute(f"UPDATE {table} SET {col} = ? WHERE rowid = ?", (new_val, rowid))
                                changes += 1
            except:
                pass

    conn.commit()
    conn.close()
    print(f"✅ Success! {changes} hidden paths fixed in the database.")

if __name__ == "__main__":
    ultimate_path_fixer()