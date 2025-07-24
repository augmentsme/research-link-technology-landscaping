The API documentation for data access is in https://researchlink.ardc.edu.au/v3/api-docs/public
The default python interpreter is in ~/.conda/envs/rapids-24.12/bin/python, do not install any package into this environment, if you need anything just add them requirements.txt. Don't run `pip install` commands, just assume all packages are already installed. 

The runCommands tool sometimes fails to capture the command output. If that happens, use the get_terminal_last_command tool to retrieve the last
command output from the terminal. If that fails, ask the user to copy-paste the output from the terminal.
