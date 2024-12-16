from django.core.management.commands.runserver import Command as RunserverCommand
import argparse
import os


class Command(RunserverCommand):
    help = 'Run the server with additional arguments for function type.'

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            '--function_type',
            type=str,
            choices=['value', 'advantage'],
            default='value',
            help="specify whether to compute 'value' or 'advantage' function.",
        )

    def handle(self, *args, **options):
        # Set the function type in environment variables
        os.environ['FUNCTION_TYPE'] = options['function_type']
        # Call the original runserver command
        super().handle(*args, **options)
