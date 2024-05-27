import time

# Файл для хранения параметров
parameters_file = r'parameters.txt'


def write_parameters(is_moving, is_rotating_left, is_rotating_right):
    with open(parameters_file, 'w') as f:
        f.write(f"{is_moving},{is_rotating_left},{is_rotating_right}")


def update_parameters():
    is_moving = False
    is_rotating_left = False
    is_rotating_right = False

    while True:
        is_moving = True
        write_parameters(is_moving, is_rotating_left, is_rotating_right)
        time.sleep(3)  # Включить движение на 3 секунды

        is_moving = False
        write_parameters(is_moving, is_rotating_left, is_rotating_right)
        time.sleep(3)  # Остановить движение на 3 секунды


if __name__ == "__main__":
    update_parameters()