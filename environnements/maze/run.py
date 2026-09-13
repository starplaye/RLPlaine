import subprocess
import time
import signal

cmd = [
    "venv/bin/python3",
    "-u", 
    "environnements/maze/maze_train.py",
    "--max_iterations", "20000",
    "-ns",
    "-m","HRL",
    "-d"
]

for _ in range(30):
    for i in range(5):
        cmd.append(str(i))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # line-buffered
        )

        # Lecture en temps réel
        try:
            for line in proc.stdout:
                print(line, end="")
            
            proc.wait()
                
        except KeyboardInterrupt:
            proc.kill()
            print("\n[STOP] Interruption")
            exit(0)

        cmd.pop()

