from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

def print_in_dark_mode(text):
    print(Back.BLACK + Fore.WHITE + text + Style.RESET_ALL)

def print_in_light_mode(text):
    print(Back.WHITE + Fore.BLACK + text + Style.RESET_ALL)

# Example usage
if __name__ == "__main__":
    dark_mode = True  # You can set this based on user preferences or the application's dark mode status

    if dark_mode:
        print_in_dark_mode("This is a dark mode message.")
    else:
        print_in_light_mode("This is a light mode message.")
        