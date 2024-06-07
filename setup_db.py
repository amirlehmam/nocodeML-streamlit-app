import yaml
from yaml.loader import SafeLoader
from werkzeug.security import generate_password_hash

# Load existing config
with open('config.yaml', 'r') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Hash the passwords
hashed_password_admin = generate_password_hash('Azerty_94400', method='pbkdf2:sha256')
hashed_password_test_user = generate_password_hash('Azerty_94', method='pbkdf2:sha256')

# Update the config with hashed passwords
config['credentials']['usernames']['admin']['password'] = hashed_password_admin
config['credentials']['usernames']['test_user']['password'] = hashed_password_test_user

# Save the updated config
with open('config.yaml', 'w') as file:
    yaml.dump(config, file)

print("Passwords hashed and config.yaml updated successfully.")
