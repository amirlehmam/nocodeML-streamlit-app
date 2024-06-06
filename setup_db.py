# setup_db.py
import sqlite3
from werkzeug.security import generate_password_hash

conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    password TEXT NOT NULL
)
''')

# Insert a user
username = 'admin'
name = 'Administrator'
password = 'password123'
hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

# Check if user already exists
c.execute('SELECT * FROM users WHERE username=?', (username,))
user = c.fetchone()

if user:
    print("User already exists.")
else:
    c.execute('INSERT INTO users (username, name, password) VALUES (?, ?, ?)', (username, name, hashed_password))
    print("User created.")

conn.commit()
conn.close()
