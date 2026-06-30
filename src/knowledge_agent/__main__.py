"""`python -m knowledge_agent` entry — same as the `kg` console script."""
import sys

from knowledge_agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
