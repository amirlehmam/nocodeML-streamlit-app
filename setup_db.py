import yaml
from yaml.loader import SafeLoader
from werkzeug.security import generate_password_hash

# Load existing config
with open('config.yaml', 'r') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Hash the password
hashed_password = generate_password_hash('admin123', method='pbkdf2:sha256')

# Update the config with hashed password
config['credentials']['usernames']['admin']['password'] = hashed_password

# Save the updated config
with open('config.yaml', 'w') as file:
    yaml.dump(config, file)
