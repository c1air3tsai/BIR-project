"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    if len(sys.argv) == 1:
        sys.argv.append('runserver')
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'biomedir.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django is not installed in this Python environment. "
            "Install the packages once with: python3 -m pip install -r requirements.txt"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
