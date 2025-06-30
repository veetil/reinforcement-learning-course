# Claude Run - Enhanced Development System

This directory contains the Claude Run system that combines SPARC methodology with iterative planning and Claude CLI SDK integration.

## Files

- `claude-run.sh` - Main script with enhanced development workflows
- `mcp.json` - MCP configuration (no export-env.sh needed)
- `.env` - Environment variables (create this file)

## Quick Start

1. **Create .env file** (required, can be empty):
```bash
touch .env
```

Or with actual environment variables:
```bash
cat > .env << 'EOF'
GITHUB_TOKEN=your-github-token
SUPABASE_ACCESS_TOKEN=your-supabase-token
PERPLEXITY_API_KEY=your-perplexity-key
FIRECRAWL_API_KEY=your-firecrawl-key
EOF
```

2. **Run the script**:
```bash
./claude-run.sh my-project requirements.md
```

## Key Features

- **Automatic .env sourcing** - No need for export-env.sh
- **Iterative workflow by default** - Better planning through refinement
- **MCP enabled by default** - With automatic npm/npx validation
- **Three workflow modes**:
  - `iterative` (default) - Multiple planning iterations with SPCT metrics
  - `sparc` - Classic SPARC methodology
  - `hybrid` - Combines both approaches

## Example Commands

```bash
# Basic usage (iterative workflow)
./claude-run.sh my-app README.md

# Agent integration mode
./claude-run.sh --mode agent-integration agent-system readme2.md

# Hybrid workflow with full analysis
./claude-run.sh --workflow hybrid --arch-diagrams --key-modules --risk-analysis project spec.md

# Skip MCP if not needed
./claude-run.sh --disable-mcp simple-project README.md

# Custom environment file
./claude-run.sh --env-file .env.production my-app README.md
```

## MCP Servers Included

- **puppeteer** - Browser automation
- **github** - GitHub API access (requires GITHUB_TOKEN)
- **perplexityai** - AI search (requires PERPLEXITY_API_KEY)
- **supabase** - Database access (requires SUPABASE_ACCESS_TOKEN)
- **firecrawl** - Web scraping (requires FIRECRAWL_API_KEY)

## Environment Variables

The script automatically sources `.env` and exports all variables for MCP servers to use. No need for wrapper scripts!

## Requirements

- Claude CLI: `npm install -g @anthropic/claude-cli`
- Node.js with npm/npx (for MCP)
- Git (for commits)
- Environment file (.env)

## Workflow Modes

### Iterative (Default)
- Creates initial plan
- Evaluates with SPCT metrics
- Refines through multiple iterations
- Best for complex projects

### SPARC
- Specification → Pseudocode → Architecture → Refinement → Completion
- Linear progression
- Best for well-defined projects

### Hybrid
- Runs SPARC first
- Then applies iterative refinement
- Maximum thoroughness
- Best for critical projects