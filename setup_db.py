from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from werkzeug.security import generate_password_hash

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    name = Column(String, nullable=False)

DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

db = SessionLocal()

# Check if the user already exists
existing_user = db.query(User).filter_by(username='admin').first()

if not existing_user:
    hashed_password = generate_password_hash('password123', method='pbkdf2:sha256')
    new_user = User(username='admin', password=hashed_password, name='Administrator')
    db.add(new_user)
    db.commit()
    print("User 'admin' added to the database.")
else:
    print("User 'admin' already exists in the database.")

db.close()
