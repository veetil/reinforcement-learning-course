Add below to claude-run.sh. Integrate is smoothly, fix it, improve writing, then udpate 

# Task 
Summarize the repository. The state of the repo is in constant flux. It is important to update CLAUDE.mds to keep the memory current. 
# Frequency
After every 100 lines of code change total across repo. 
# Method
Traverse repository file structure like a graph. Do this recursively bottom up. Nodes added to queue as soon as their children analyzed or if they are leaf nodes. Queue nodes analyzed with max parallelism 10. 

# Details 
Parallel agents- 
Launch parallel Task / Agents to run below in parallel 
1. Parallelism - Think of the source code as graph. Do bottom up traversal. Get list of leaf nodes with no unfinished dependencies - ie, leaf sub folders and ( after initial iteration) sub folders with all leaf sub folders /children analyzed. Assign  one Agent / Task to one node. Total max 10 . Each agent analyzes the sub folder and its children, then summarizes it . Include the specs, architecture, key modules, risk factors, potential issues for each sub folder in its CLAUDE.md. Once all are done, repeat. 
Hierarchically move up till root folder and make sure all CLAUDE.md files are updated. Parents will use child CLAUDE.md and individual file anlaysis to create their CLAUDE.md
