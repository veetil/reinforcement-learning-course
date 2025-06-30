#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Get the directory where this script is located
const scriptDir = __dirname;
const bashScriptPath = path.join(scriptDir, 'claude-run.sh');
const mpcConfigPath = path.join(scriptDir, 'mcp.json');

// Get command line arguments (skip first two: node and script path)
const args = process.argv.slice(2);

// Handle init command specially
if (args[0] === 'init') {
  console.log('Initializing Claude Run CLI project...');
  
  // Create .env file if it doesn't exist
  if (!fs.existsSync('.env')) {
    fs.writeFileSync('.env', `# Environment variables for Claude Run CLI
# Add your tokens here:
# GITHUB_TOKEN=your-github-token
# SUPABASE_ACCESS_TOKEN=your-supabase-token
# PERPLEXITY_API_KEY=your-perplexity-key
# FIRECRAWL_API_KEY=your-firecrawl-key
`);
    console.log('✓ Created .env file');
  }
  
  // Copy mcp.json to current directory
  if (!fs.existsSync('mcp.json')) {
    const mpcContent = fs.readFileSync(mpcConfigPath, 'utf8');
    fs.writeFileSync('mcp.json', mpcContent);
    console.log('✓ Created mcp.json file');
  }
  
  // Create a sample requirements.md
  if (!fs.existsSync('requirements.md')) {
    fs.writeFileSync('requirements.md', `# Project Requirements

## Overview
Describe your project here...

## Features
- Feature 1
- Feature 2
- Feature 3

## Technical Requirements
- Technology stack
- Performance requirements
- Security requirements

## Success Criteria
- What defines success for this project?
`);
    console.log('✓ Created requirements.md template');
  }
  
  // Create CLAUDE.md for project context
  if (!fs.existsSync('CLAUDE.md')) {
    fs.writeFileSync('CLAUDE.md', `# Claude Project Context

This file maintains context for Claude about your project.

## Project Structure
- Document your project structure here

## Architecture Decisions
- List key architectural decisions

## Development Guidelines
- Coding standards
- Testing approach
- Documentation requirements
`);
    console.log('✓ Created CLAUDE.md file');
  }
  
  console.log('\nInitialization complete! Next steps:');
  console.log('1. Edit requirements.md to describe your project');
  console.log('2. Add any necessary tokens to .env');
  console.log('3. Run: npx claude-run-cli requirements.md');
  process.exit(0);
}

// Check if bash script exists
if (!fs.existsSync(bashScriptPath)) {
  console.error('Error: claude-run.sh not found!');
  console.error('Script directory:', scriptDir);
  console.error('Looking for:', bashScriptPath);
  process.exit(1);
}

// Make sure the bash script is executable
try {
  fs.chmodSync(bashScriptPath, '755');
} catch (err) {
  console.error('Error: Could not make claude-run.sh executable');
  process.exit(1);
}

// If no arguments provided, show interactive prompt
if (args.length === 0) {
  console.log(`
Claude Run CLI - Enhanced Development System
===========================================

This system combines SPARC methodology with iterative planning to help you
build high-quality software with Claude's assistance.

Available workflows:
- SPARC: Specification → Pseudocode → Architecture → Refinement → Completion
- Iterative: Multiple planning rounds with metrics-based refinement
- Hybrid: Combines both approaches for maximum thoroughness

To get started:
1. Create a requirements file: npx claude-run-cli init
2. Run with your requirements: npx claude-run-cli requirements.md
3. Or specify any markdown file: npx claude-run-cli your-spec.md

For all options: npx claude-run-cli --help

What would you like to do?
- Type a filename to use as requirements (e.g., spec.md)
- Type 'init' to create starter files
- Type 'help' for more information
- Press Ctrl+C to exit
`);
  
  // Set up stdin for interactive input
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', (input) => {
    const userInput = input.trim();
    
    if (userInput === 'init') {
      // Re-run with init argument
      process.argv.push('init');
      require(process.argv[1]);
    } else if (userInput === 'help') {
      // Run with --help
      spawnBashScript(['--help']);
    } else if (userInput) {
      // Use the input as a filename
      spawnBashScript([userInput]);
    }
  });
  
  return;
}

// If user requests help about npx specifically
if (args.includes('--npx-help')) {
  console.log('Claude Run CLI - NPX Usage');
  console.log('========================');
  console.log('');
  console.log('This is the npx wrapper for claude-run.');
  console.log('');
  console.log('Usage:');
  console.log('  npx claude-run-cli init              - Create starter files');
  console.log('  npx claude-run-cli [options] [file]   - Run with requirements file');
  console.log('  npx claude-run-cli                    - Interactive mode');
  console.log('');
  console.log('The wrapper executes the enhanced claude-run.sh script with:');
  console.log('- Automatic MCP configuration from included mcp.json');
  console.log('- All workflow modes (sparc, iterative, hybrid)');
  console.log('- Parallel execution optimization');
  console.log('- Hierarchical documentation updates');
  console.log('');
  console.log('For full options, run:');
  console.log('  npx claude-run-cli --help');
  console.log('');
  process.exit(0);
}

// Function to spawn the bash script
function spawnBashScript(additionalArgs = []) {
  // Check if MCP config should be used (default behavior)
  let mpcConfigArgs = [];
  const allArgs = [...args, ...additionalArgs];
  
  if (!allArgs.includes('--disable-mcp') && fs.existsSync(mpcConfigPath)) {
    // Add the bundled MCP config if not explicitly disabled
    if (!allArgs.some(arg => arg === '-c' || arg === '--config')) {
      mpcConfigArgs = ['-c', mpcConfigPath];
    }
  }
  
  // Combine args: MCP config args + user args
  const finalArgs = [...mpcConfigArgs, ...allArgs];
  
  console.log('Initializing Claude Run CLI...');
  
  // Spawn the bash script with arguments
  const child = spawn('/bin/bash', [bashScriptPath, ...finalArgs], {
    stdio: 'inherit',
    shell: false,
    env: { ...process.env }
  });
  
  // Handle errors
  child.on('error', (error) => {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  });
  
  // Exit with the same code as the bash script
  child.on('exit', (code) => {
    process.exit(code || 0);
  });
}

// Normal execution - pass through to bash script
spawnBashScript();