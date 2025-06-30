#!/bin/bash

# Claude Run - Enhanced Development System
# Combines SPARC methodology with iterative planning and Claude CLI SDK integration
# Supports both automated development workflows and agent system integration

set -e  # Exit on any error

# Default configuration
PROJECT_NAME="claude-project"
README_PATH="README.md"
MCP_CONFIG="./mcp.json"
VERBOSE=false
DRY_RUN=false
SKIP_RESEARCH=false
SKIP_TESTS=false
TEST_COVERAGE_TARGET=100
PARALLEL_EXECUTION=true
COMMIT_FREQUENCY="phase"  # phase, feature, or manual
OUTPUT_FORMAT="text"
DEVELOPMENT_MODE="full"   # full, backend-only, frontend-only, api-only, agent-integration
WORKFLOW_MODE="iterative"     # sparc, iterative, hybrid
AGENT_SDK_PATH="claude-agent-system"
GITHUB_REPO=""
ITERATION_COUNT=3
METRICS_ENABLED=true
ARCHITECTURE_DIAGRAMS=true
ENABLE_MCP=true
ENV_FILE=".env"
MAX_PARALLELISM=10  # Global max parallel tasks
FORCE_PARALLEL=true  # Always seek parallelism opportunities

# Help function
show_help() {
    cat << EOF
Claude Run - Enhanced Development System
=======================================

A comprehensive development system combining SPARC methodology with iterative planning,
Claude CLI SDK integration, and advanced workflow management.

USAGE:
    ./claude-run.sh [OPTIONS] [PROJECT_NAME] [README_PATH]

ARGUMENTS:
    PROJECT_NAME    Name of the project to develop (default: claude-project)
    README_PATH     Path to requirements file (default: README.md, auto-searches for common names)

OPTIONS:
    -h, --help                  Show this help message
    -v, --verbose              Enable verbose output
    -d, --dry-run              Show what would be done without executing
    -c, --config FILE          MCP configuration file (default: ./mcp.json)
    
    # Workflow Options
    --workflow MODE            Workflow mode: sparc, iterative, hybrid (default: iterative)
    --iterations COUNT         Number of planning iterations (default: 3)
    --metrics                  Enable SPCT metrics evaluation (default: true)
    --no-metrics               Disable metrics evaluation
    
    # Research Options
    --skip-research            Skip the web research phase
    --research-depth LEVEL     Research depth: basic, standard, comprehensive (default: standard)
    
    # Development Options
    --mode MODE                Development mode: full, backend-only, frontend-only, api-only, agent-integration (default: full)
    --skip-tests               Skip test development (not recommended)
    --coverage TARGET          Test coverage target percentage (default: 100)
    --no-parallel              Disable parallel execution
    --max-parallel COUNT       Maximum parallel tasks globally (default: 10)
    --force-parallel           Always seek parallelism opportunities (default: true)
    
    # Agent Integration Options
    --agent-sdk PATH           Path to Claude agent SDK system (default: claude-agent-system)
    --enable-mcp               Enable MCP support in agent integration (default: true)
    --disable-mcp              Disable MCP support
    --env-file FILE            Environment file to source (default: .env)
    --input-mapping            Enable input file/folder mapping for agent system
    
    # Architecture Options
    --arch-diagrams            Generate architecture diagrams (default: true)
    --key-modules              Generate KEY_MODULES.md analysis
    --risk-analysis            Generate RISK_FACTORS.md analysis
    
    # Documentation Update Options
    --doc-update               Enable hierarchical CLAUDE.md updates (default: true)
    --no-doc-update            Disable hierarchical CLAUDE.md updates
    --doc-threshold LINES      Lines of code change threshold for doc updates (default: 100)
    --doc-parallelism COUNT    Max parallel agents for doc updates (default: 10)
    
    # Commit Options
    --commit-freq FREQ         Commit frequency: phase, feature, manual (default: phase)
    --no-commits               Disable automatic commits
    --github-repo REPO         GitHub repository for commits (format: owner/repo)
    
    # Output Options
    --output FORMAT            Output format: text, json, markdown (default: text)
    --quiet                    Suppress non-essential output

WORKFLOW MODES:
    sparc                Classic SPARC methodology (Specification, Pseudocode, Architecture, Refinement, Completion)
    iterative            Iterative planning with metrics evaluation and refinement (default)
    hybrid               Combines SPARC with iterative planning and comprehensive analysis

DEVELOPMENT MODES:
    full                 Complete full-stack development (default)
    backend-only         Backend services and APIs only
    frontend-only        Frontend application only
    api-only             REST/GraphQL API development only
    agent-integration    Claude CLI SDK agent system integration

EXAMPLES:
    # Basic SPARC development
    ./claude-run.sh my-app requirements.md
    
    # Using a custom requirements file
    ./claude-run.sh implement-this.md
    
    # Agent integration with iterative planning
    ./claude-run.sh --workflow iterative --mode agent-integration --enable-mcp agent-app readme2.md
    
    # Hybrid workflow with architecture analysis
    ./claude-run.sh --workflow hybrid --arch-diagrams --key-modules --risk-analysis my-project spec.md
    
    # Full development with GitHub integration
    ./claude-run.sh --github-repo myuser/myproject --iterations 5 full-app requirements.md
    
    # Development with custom documentation update settings
    ./claude-run.sh --doc-threshold 50 --doc-parallelism 15 my-project spec.md

PHASES:
    1. Research & Discovery     - Comprehensive domain and technology research
    2. Specification           - Requirements analysis and system specification
    3. Architecture            - System design with diagrams and analysis
    4. Iterative Planning      - Multi-round planning with metrics evaluation
    5. Implementation          - TDD-based incremental development
    6. Integration & Testing   - Comprehensive testing and quality assurance
    7. Deployment             - Production readiness and deployment

For more information, see claude-run-guide.md
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -c|--config)
                MCP_CONFIG="$2"
                shift 2
                ;;
            --workflow)
                WORKFLOW_MODE="$2"
                shift 2
                ;;
            --iterations)
                ITERATION_COUNT="$2"
                shift 2
                ;;
            --metrics)
                METRICS_ENABLED=true
                shift
                ;;
            --no-metrics)
                METRICS_ENABLED=false
                shift
                ;;
            --skip-research)
                SKIP_RESEARCH=true
                shift
                ;;
            --research-depth)
                RESEARCH_DEPTH="$2"
                shift 2
                ;;
            --mode)
                DEVELOPMENT_MODE="$2"
                shift 2
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --coverage)
                TEST_COVERAGE_TARGET="$2"
                shift 2
                ;;
            --no-parallel)
                PARALLEL_EXECUTION=false
                FORCE_PARALLEL=false
                shift
                ;;
            --max-parallel)
                MAX_PARALLELISM="$2"
                shift 2
                ;;
            --force-parallel)
                FORCE_PARALLEL=true
                shift
                ;;
            --agent-sdk)
                AGENT_SDK_PATH="$2"
                shift 2
                ;;
            --enable-mcp)
                ENABLE_MCP=true
                shift
                ;;
            --disable-mcp)
                ENABLE_MCP=false
                shift
                ;;
            --env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            --input-mapping)
                INPUT_MAPPING=true
                shift
                ;;
            --arch-diagrams)
                ARCHITECTURE_DIAGRAMS=true
                shift
                ;;
            --key-modules)
                KEY_MODULES_ANALYSIS=true
                shift
                ;;
            --risk-analysis)
                RISK_ANALYSIS=true
                shift
                ;;
            --doc-update)
                DOCUMENTATION_UPDATE_ENABLED=true
                shift
                ;;
            --no-doc-update)
                DOCUMENTATION_UPDATE_ENABLED=false
                shift
                ;;
            --doc-threshold)
                DOCUMENTATION_UPDATE_THRESHOLD="$2"
                shift 2
                ;;
            --doc-parallelism)
                DOCUMENTATION_MAX_PARALLELISM="$2"
                shift 2
                ;;
            --commit-freq)
                COMMIT_FREQUENCY="$2"
                shift 2
                ;;
            --no-commits)
                COMMIT_FREQUENCY="manual"
                shift
                ;;
            --github-repo)
                GITHUB_REPO="$2"
                shift 2
                ;;
            --output)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            --quiet)
                VERBOSE=false
                shift
                ;;
            -*)
                echo "Unknown option: $1" >&2
                echo "Use --help for usage information" >&2
                exit 1
                ;;
            *)
                # If the argument looks like a file (has an extension), treat it as README_PATH
                if [[ "$1" == *.* ]] && [[ -f "$1" || ! -e "$1" ]]; then
                    # It's likely a file, use it as README_PATH
                    if [[ "$README_PATH" == "README.md" ]]; then
                        README_PATH="$1"
                        # If project name is still default, derive it from the file
                        if [[ "$PROJECT_NAME" == "claude-project" ]]; then
                            PROJECT_NAME="${1%.*}"  # Remove extension
                            PROJECT_NAME="${PROJECT_NAME##*/}"  # Remove path
                        fi
                    else
                        echo "Too many arguments: $1" >&2
                        echo "Use --help for usage information" >&2
                        exit 1
                    fi
                else
                    # No extension or is a directory, treat as project name
                    if [[ "$PROJECT_NAME" == "claude-project" ]]; then
                        PROJECT_NAME="$1"
                    elif [[ "$README_PATH" == "README.md" ]]; then
                        README_PATH="$1"
                    else
                        echo "Too many arguments: $1" >&2
                        echo "Use --help for usage information" >&2
                        exit 1
                    fi
                fi
                shift
                ;;
        esac
    done
}

# Validate configuration
validate_config() {
    # Check for ANTHROPIC_API_KEY and unset with warning
    if [[ -n "$ANTHROPIC_API_KEY" ]]; then
        echo "Warning: ANTHROPIC_API_KEY environment variable detected." >&2
        echo "Unsetting ANTHROPIC_API_KEY to use claude-cli authentication instead." >&2
        echo "The CLI uses its own authentication mechanism." >&2
        unset ANTHROPIC_API_KEY
    fi
    
    # Check for environment file (required whether default .env or custom specified)
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "Error: Environment file not found: $ENV_FILE" >&2
        echo "Please create an environment file (can be empty):" >&2
        echo "  touch $ENV_FILE" >&2
        echo "Or specify a different file with: --env-file /path/to/.env" >&2
        exit 1
    fi
    
    # Source environment file
    if [[ "$VERBOSE" == true ]]; then
        echo "Sourcing environment file: $ENV_FILE" >&2
    fi
    # Export all variables from the env file
    set -a
    source "$ENV_FILE"
    set +a
    
    # Check if MCP config exists when MCP is enabled
    if [[ "$ENABLE_MCP" == true ]]; then
        # Check for npm and npx
        if ! command -v npm &> /dev/null; then
            echo "Error: npm is required for MCP but not found in PATH" >&2
            echo "Please install Node.js and npm: https://nodejs.org/" >&2
            exit 1
        fi
        
        if ! command -v npx &> /dev/null; then
            echo "Error: npx is required for MCP but not found in PATH" >&2
            echo "Please ensure npm is properly installed with npx" >&2
            exit 1
        fi
        
        if [[ ! -f "$MCP_CONFIG" ]]; then
            echo "Error: MCP is enabled but config file not found: $MCP_CONFIG" >&2
            echo "Please either:" >&2
            echo "  1. Create the MCP config file at: $MCP_CONFIG" >&2
            echo "  2. Specify a different config with: --config /path/to/mcp.json" >&2
            echo "  3. Disable MCP with: --disable-mcp" >&2
            echo "" >&2
            echo "Example MCP config structure:" >&2
            echo '{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
    }
  }
}' >&2
            exit 1
        fi
    fi
    
    # Check if requirements file exists
    if [[ ! -f "$README_PATH" ]]; then
        # Only search for README alternatives if the user didn't specify a custom file
        if [[ "$README_PATH" == "README.md" ]]; then
            # User didn't specify a file, so try common README file variations
            local readme_alternatives=("README.md" "readme.md" "Readme.md" "README.txt" "readme.txt" "readme2.md")
            local found_readme=""
            
            for alt in "${readme_alternatives[@]}"; do
                if [[ -f "$alt" ]]; then
                    found_readme="$alt"
                    break
                fi
            done
            
            if [[ -n "$found_readme" ]]; then
                echo "README file '$README_PATH' not found, using '$found_readme' instead" >&2
                README_PATH="$found_readme"
            else
                echo "Error: No README file found. Tried: ${readme_alternatives[*]}" >&2
                echo "Please specify a valid requirements file path or create one of the above files." >&2
                exit 1
            fi
        else
            # User specified a custom file that doesn't exist
            echo "Error: Specified requirements file not found: $README_PATH" >&2
            echo "Please check the file path and try again." >&2
            exit 1
        fi
    fi
    
    # Validate workflow mode
    case $WORKFLOW_MODE in
        sparc|iterative|hybrid) ;;
        *) echo "Error: Invalid workflow mode: $WORKFLOW_MODE" >&2; exit 1 ;;
    esac
    
    # Validate development mode
    case $DEVELOPMENT_MODE in
        full|backend-only|frontend-only|api-only|agent-integration) ;;
        *) echo "Error: Invalid development mode: $DEVELOPMENT_MODE" >&2; exit 1 ;;
    esac
    
    # Validate commit frequency
    case $COMMIT_FREQUENCY in
        phase|feature|manual) ;;
        *) echo "Error: Invalid commit frequency: $COMMIT_FREQUENCY" >&2; exit 1 ;;
    esac
    
    # Validate output format
    case $OUTPUT_FORMAT in
        text|json|markdown) ;;
        *) echo "Error: Invalid output format: $OUTPUT_FORMAT" >&2; exit 1 ;;
    esac
    
    # Validate coverage target
    if [[ ! "$TEST_COVERAGE_TARGET" =~ ^[0-9]+$ ]] || [[ "$TEST_COVERAGE_TARGET" -lt 0 ]] || [[ "$TEST_COVERAGE_TARGET" -gt 100 ]]; then
        echo "Error: Invalid coverage target: $TEST_COVERAGE_TARGET (must be 0-100)" >&2
        exit 1
    fi
    
    # Validate iteration count
    if [[ ! "$ITERATION_COUNT" =~ ^[0-9]+$ ]] || [[ "$ITERATION_COUNT" -lt 1 ]] || [[ "$ITERATION_COUNT" -gt 10 ]]; then
        echo "Error: Invalid iteration count: $ITERATION_COUNT (must be 1-10)" >&2
        exit 1
    fi
    
    # Validate documentation update threshold
    if [[ ! "$DOCUMENTATION_UPDATE_THRESHOLD" =~ ^[0-9]+$ ]] || [[ "$DOCUMENTATION_UPDATE_THRESHOLD" -lt 1 ]]; then
        echo "Error: Invalid documentation update threshold: $DOCUMENTATION_UPDATE_THRESHOLD (must be positive integer)" >&2
        exit 1
    fi
    
    # Validate documentation update parallelism
    if [[ ! "$DOCUMENTATION_MAX_PARALLELISM" =~ ^[0-9]+$ ]] || [[ "$DOCUMENTATION_MAX_PARALLELISM" -lt 1 ]] || [[ "$DOCUMENTATION_MAX_PARALLELISM" -gt 20 ]]; then
        echo "Error: Invalid documentation update parallelism: $DOCUMENTATION_MAX_PARALLELISM (must be 1-20)" >&2
        exit 1
    fi
    
    # Check agent SDK path if agent-integration mode
    if [[ "$DEVELOPMENT_MODE" == "agent-integration" ]] && [[ ! -d "$AGENT_SDK_PATH" ]]; then
        echo "Warning: Agent SDK path not found: $AGENT_SDK_PATH" >&2
        echo "Agent integration features may be limited" >&2
    fi
}

# Show configuration
show_config() {
    if [[ "$VERBOSE" == true ]]; then
        cat << EOF
Claude Run Configuration:
========================
Project Name: $PROJECT_NAME
README Path: $README_PATH
MCP Config: $MCP_CONFIG
Workflow Mode: $WORKFLOW_MODE
Development Mode: $DEVELOPMENT_MODE
Iterations: $ITERATION_COUNT
Research Depth: ${RESEARCH_DEPTH:-standard}
Test Coverage Target: $TEST_COVERAGE_TARGET%
Parallel Execution: $PARALLEL_EXECUTION
Force Parallel: $FORCE_PARALLEL
Max Parallelism: $MAX_PARALLELISM agents
Commit Frequency: $COMMIT_FREQUENCY
GitHub Repo: ${GITHUB_REPO:-none}
Output Format: $OUTPUT_FORMAT
Metrics Enabled: $METRICS_ENABLED
Architecture Diagrams: $ARCHITECTURE_DIAGRAMS
Documentation Updates: $DOCUMENTATION_UPDATE_ENABLED
Doc Update Threshold: $DOCUMENTATION_UPDATE_THRESHOLD lines
Doc Update Parallelism: $DOCUMENTATION_MAX_PARALLELISM agents
MCP Enabled: $ENABLE_MCP
Skip Research: $SKIP_RESEARCH
Skip Tests: $SKIP_TESTS
Dry Run: $DRY_RUN
========================
EOF
    fi
}

# Build allowed tools based on configuration
build_allowed_tools() {
    local tools="View,Edit,Replace,GlobTool,GrepTool,LS,Bash"
    
    if [[ "$SKIP_RESEARCH" != true ]]; then
        tools="$tools,WebFetchTool"
    fi
    
    if [[ "$PARALLEL_EXECUTION" == true ]]; then
        tools="$tools,BatchTool,dispatch_agent"
    fi
    
    if [[ -n "$GITHUB_REPO" ]]; then
        tools="$tools,mcp__github__create_repository,mcp__github__create_or_update_file,mcp__github__push_files"
    fi
    
    if [[ "$DEVELOPMENT_MODE" == "agent-integration" ]] && [[ "$ENABLE_MCP" == true ]]; then
        tools="$tools,mcp__puppeteer__puppeteer_navigate,mcp__puppeteer__puppeteer_screenshot"
    fi
    
    echo "$tools"
}

# Build Claude command flags
build_claude_flags() {
    local flags="--mcp-config $MCP_CONFIG --dangerously-skip-permissions"
    
    if [[ "$VERBOSE" == true ]]; then
        flags="$flags --verbose"
    fi
    
    if [[ "$OUTPUT_FORMAT" != "text" ]]; then
        flags="$flags --output-format $OUTPUT_FORMAT"
    fi
    
    echo "$flags"
}

# Generate workflow prompt based on mode
generate_workflow_prompt() {
    local prompt=""
    
    # Header
    prompt+="# Claude Run - Enhanced Development System\n"
    prompt+="# Project: ${PROJECT_NAME}\n"
    prompt+="# Requirements: ${README_PATH}\n"
    prompt+="# Workflow: ${WORKFLOW_MODE}\n"
    prompt+="# Mode: ${DEVELOPMENT_MODE}\n"
    
    # Include research phase unless skipped
    if [[ "$SKIP_RESEARCH" != true ]]; then
        prompt+="\n$(generate_research_phase)"
    fi
    
    # Include workflow-specific phases
    case $WORKFLOW_MODE in
        sparc)
            prompt+="\n$(generate_sparc_phases)"
            ;;
        iterative)
            prompt+="\n$(generate_iterative_phases)"
            ;;
        hybrid)
            prompt+="\n$(generate_hybrid_phases)"
            ;;
    esac
    
    # Include implementation phase
    prompt+="\n$(generate_implementation_phase)"
    
    # Include quality standards
    prompt+="\n$(generate_quality_standards)"
    
    # Include parallel execution principles if force parallel is enabled
    if [[ "$FORCE_PARALLEL" == true ]]; then
        prompt+="\n$(generate_parallel_principles)"
    fi
    
    # Include documentation update phase if enabled
    if [[ "$DOCUMENTATION_UPDATE_ENABLED" == true ]]; then
        prompt+="\n$(generate_documentation_update_phase)"
    fi
    
    echo -e "$prompt"
}

# Generate research phase
generate_research_phase() {
    cat << 'RESEARCH_PHASE'
## PHASE 0: COMPREHENSIVE PARALLEL RESEARCH & DISCOVERY
### Parallel Execution Strategy: ${MAX_PARALLELISM} concurrent tasks
### Research Depth: ${RESEARCH_DEPTH:-standard}

### Parallel Research Execution:

**Stage 1: Initial Parallel Discovery** (${MAX_PARALLELISM} concurrent agents)
Launch parallel Task agents for:
1. Domain analysis from ${README_PATH}
2. Technology landscape research
3. Competitive analysis
4. Best practices investigation
5. Security & compliance research
6. Testing framework analysis
7. Deployment pattern research
8. Performance optimization strategies
9. Monitoring solutions research
10. Architecture pattern discovery

**Stage 2: Deep-Dive Parallel Research** (Based on Stage 1 results)
Analyze Stage 1 results, then launch parallel deep-dives:
- For each identified technology: research implementation details
- For each architectural pattern: find concrete examples
- For each security concern: investigate mitigation strategies
- For each performance consideration: research optimization techniques

$(if [[ "$DEVELOPMENT_MODE" == "agent-integration" ]]; then cat << 'AGENT_RESEARCH'
**Stage 3: Agent-Specific Parallel Research**
- Claude CLI SDK capabilities analysis
- MCP integration pattern research
- Multi-agent orchestration strategies
- Parallel workflow management techniques
AGENT_RESEARCH
fi)

### Parallel Results Synthesis:
1. Gather all parallel research results
2. Identify common patterns and conflicts
3. Create unified technology recommendations
4. Generate prioritized implementation approach

**Commit**: 'feat: complete parallel research - ${MAX_PARALLELISM} concurrent investigations synthesized'
RESEARCH_PHASE
}

# Generate SPARC phases
generate_sparc_phases() {
    cat << 'SPARC_PHASES'
## SPECIFICATION PHASE - PARALLEL DECOMPOSITION
### Parallel Specification Strategy:

**Stage 1: High-Level Specification Outline** (Single task)
- Analyze ${README_PATH} to create system overview
- Identify major components and subsystems
- Define component boundaries for parallel analysis

**Stage 2: Parallel Component Specification** (${MAX_PARALLELISM} agents)
Launch parallel Task agents to specify each component:
- Component 1: API specification (endpoints, models, auth)
- Component 2: Frontend specification (UI/UX, workflows)
- Component 3: Data layer specification (schemas, relationships)
- Component 4: Integration specification (external services)
- Component 5: Security specification (auth, encryption)
- Component 6: Performance specification (metrics, SLAs)
- Component 7: Infrastructure specification (deployment)
- Component 8: Testing specification (strategies, coverage)
- Component 9: Monitoring specification (observability)
- Component 10: Documentation specification (user/dev docs)

**Stage 3: Parallel Deep-Dive Specifications**
For each component from Stage 2, parallelize sub-specifications:
- API: Parallelize endpoint groups (user, admin, public, etc.)
- Frontend: Parallelize page/feature specifications
- Data: Parallelize domain model specifications
- Security: Parallelize threat model components

**Stage 4: Specification Integration**
- Gather all parallel specifications
- Resolve cross-component dependencies
- Create unified specification document
- Validate completeness and consistency

**Commit**: 'docs: complete parallel specification - ${MAX_PARALLELISM} concurrent component specs integrated'

## PSEUDOCODE PHASE - PARALLEL ALGORITHM DESIGN
### Parallel Algorithm Development:

**Stage 1: Component Interface Design** (${MAX_PARALLELISM} agents)
Parallel Task agents design interfaces for each component:
- API contracts and protocols
- Frontend component interfaces
- Data access layer interfaces
- Service communication protocols
- Event/message schemas

**Stage 2: Parallel Algorithm Implementation** (per component)
For each component, parallelize algorithm design:
- Business logic algorithms (parallel by domain)
- Data transformation pipelines (parallel by data type)
- Validation and security checks (parallel by component)
- Performance optimization strategies (parallel by bottleneck)

**Stage 3: Test Strategy Parallelization**
- Unit test design per component (parallel)
- Integration test scenarios (parallel by interaction)
- E2E test workflows (parallel by user journey)
- Performance test suites (parallel by load type)

**Commit**: 'design: complete parallel pseudocode - ${MAX_PARALLELISM} concurrent algorithm designs'

## ARCHITECTURE PHASE - PARALLEL DEEP DESIGN
### Parallel Architecture Generation:

**Stage 1: Architecture Decomposition**
- Create high-level architecture outline
- Identify architectural components for parallel design
- Define integration boundaries

**Stage 2: Parallel Architecture Design** (${MAX_PARALLELISM} agents)
$(if [[ "$ARCHITECTURE_DIAGRAMS" == true ]]; then cat << 'ARCH_PARALLEL'
Launch parallel agents for architectural diagrams:
- System architecture diagram
- Component interaction diagrams
- Data flow diagrams
- Deployment architecture
- Security architecture
- Network topology
- CI/CD pipeline architecture
- Monitoring architecture
ARCH_PARALLEL
fi)

$(if [[ "$KEY_MODULES_ANALYSIS" == true ]]; then cat << 'KEY_MODULES_PARALLEL'
**Stage 3: Parallel Key Modules Analysis**
- Identify top-level critical modules
- Launch parallel analysis for each module:
  * Module specifications
  * Success factors
  * Dependencies
  * Risk factors
  * Performance characteristics
KEY_MODULES_PARALLEL
fi)

$(if [[ "$RISK_ANALYSIS" == true ]]; then cat << 'RISK_PARALLEL'
**Stage 4: Parallel Risk Analysis**
- Identify risk categories
- Parallel risk assessment per category:
  * Technical risks
  * Security risks
  * Performance risks
  * Scalability risks
  * Integration risks
  * Operational risks
RISK_PARALLEL
fi)

**Stage 5: Architecture Integration**
- Synthesize all parallel architectural designs
- Resolve conflicts and dependencies
- Create unified ARCHITECTURE.md
- Validate architectural consistency

**Commit**: 'arch: complete parallel architecture - ${MAX_PARALLELISM} concurrent designs integrated'

## REFINEMENT PHASE - PARALLEL TDD IMPLEMENTATION
### Parallel Development Strategy:
Create folder structure for parallel execution:
- 1_parallel/: Independent modules (auth, user, admin, api)
- 2_parallel/: Service layers (business, data, integration)
- 3_parallel/: Frontend components (pages, widgets, utils)
- 4_integration/: Sequential integration phase

Each parallel folder contains tasks that can execute concurrently.
SPARC_PHASES
}

# Generate iterative phases
generate_iterative_phases() {
    cat << 'ITERATIVE_PHASES'
## ITERATIVE PLANNING PHASE - MAXIMIZING PARALLELISM
### Parallel Planning Strategy:

**Initial Planning - Parallel Decomposition**:
1. **System Analysis** (Single task):
   - Extract high-level requirements from ${README_PATH}
   - Identify independent subsystems for parallel planning
   - Define integration points between subsystems

2. **Parallel Subsystem Planning** (${MAX_PARALLELISM} agents):
   Launch parallel Task agents to plan each subsystem:
   - Frontend planning (components, routes, state)
   - Backend API planning (endpoints, middleware, auth)
   - Data layer planning (models, migrations, queries)
   - Infrastructure planning (deployment, scaling, monitoring)
   - Testing planning (unit, integration, e2e strategies)
   - Security planning (auth, encryption, validation)
   - Performance planning (caching, optimization, CDN)
   - Documentation planning (user guides, API docs)

3. **Parallel Step Organization**:
   Create folder structure emphasizing parallelism:
   ```
   plan_r${iteration}/
   ├── 1_parallel/     # Independent foundation tasks
   │   ├── setup_dev_env.md
   │   ├── create_project_structure.md
   │   ├── configure_tools.md
   │   └── initialize_repos.md
   ├── 2_parallel/     # Core module development
   │   ├── auth_module.md
   │   ├── user_module.md
   │   ├── data_models.md
   │   └── api_framework.md
   ├── 3_parallel/     # Feature development
   │   ├── frontend_components.md
   │   ├── backend_services.md
   │   ├── database_layer.md
   │   └── external_integrations.md
   └── 4_sequential/   # Integration only
       └── system_integration.md
   ```

### Iterative Refinement (${ITERATION_COUNT} iterations):
For iteration in 1..${ITERATION_COUNT}:
   1. **Parallel Evaluation** (${MAX_PARALLELISM} agents):
      - Each agent evaluates different aspects:
        * SPCT metrics evaluation
        * Dependency analysis
        * Risk assessment
        * Resource optimization
        * Parallelism opportunities
   
   2. **Parallel Deep Research**:
      Based on gaps found, launch parallel research:
      - Technology-specific best practices
      - Performance optimization techniques
      - Security hardening approaches
      - Testing methodology improvements
   
   3. **Parallel Plan Refinement**:
      - Each subsystem refined by dedicated agent
      - Maximize parallel execution opportunities
      - Minimize sequential bottlenecks
      - Optimize resource utilization
   
   4. **Integration Validation**:
      - Verify parallel steps don't conflict
      - Ensure clean integration points
      - Validate end-to-end workflows
      - Confirm parallel execution feasibility

### Final Parallel Plan:
- Structure: plan_final/ with maximized parallelism
- Each step annotated with:
  * Dependencies (what must complete first)
  * Outputs (what this step produces)
  * Integration points (how results combine)
  * Estimated parallel execution time
- Clear integration checkpoints between parallel phases

**Commit**: 'plan: iterative planning optimized for ${MAX_PARALLELISM}-way parallelism'
ITERATIVE_PHASES
}

# Generate hybrid phases
generate_hybrid_phases() {
    cat << 'HYBRID_PHASES'
## HYBRID WORKFLOW: SPARC + ITERATIVE PLANNING

### Phase 1: SPARC Foundation
$(generate_sparc_phases)

### Phase 2: Iterative Enhancement
$(generate_iterative_phases)

### Phase 3: Integration & Synthesis
1. **Architecture Refinement**:
   - Merge SPARC architecture with iterative insights
   - Update ARCHITECTURE.md with comprehensive diagrams
   - Refine component interactions based on planning
   - Optimize for identified critical success factors

2. **Risk Mitigation**:
   - Address all identified risks from both methodologies
   - Create contingency plans for high-risk areas
   - Design fallback strategies for critical components
   - Update RISK_FACTORS.md with mitigation strategies

3. **Module Optimization**:
   - Focus development on key modules from KEY_MODULES.md
   - Prioritize critical success factors
   - Design for maximum impact on project outcomes
   - Create detailed implementation guides

**Commit**: 'arch: complete hybrid workflow synthesis - integrated SPARC and iterative methodologies'
HYBRID_PHASES
}

# Generate implementation phase
generate_implementation_phase() {
    cat << 'IMPLEMENTATION_PHASE'
## IMPLEMENTATION PHASE - MASSIVE PARALLELIZATION
### Parallel-First Development Strategy:

**Phase 1: Parallel Foundation** (1_parallel/)
Launch ${MAX_PARALLELISM} agents concurrently for:
- Development environment setup
- Project scaffolding (frontend, backend, infra)
- Tool configuration (linters, formatters, CI)
- Repository initialization
- Dependency installation
- Database setup
- Container configuration
- Testing framework setup

**Phase 2: Parallel Module Development** (2_parallel/)
For each independent module, launch parallel agents:

*Backend Modules* (parallel execution):
- Auth service (JWT, OAuth, sessions)
- User management service
- Core business logic services
- Data access layer
- API gateway setup
- Middleware components
- Background job processors
- Cache layer implementation

*Frontend Modules* (parallel execution):
- Component library setup
- Page components
- State management
- Routing configuration
- API client services
- Utility functions
- Style system
- Asset management

*Infrastructure Modules* (parallel execution):
- Docker configurations
- Kubernetes manifests
- CI/CD pipelines
- Monitoring setup
- Logging configuration
- Security policies
- Backup strategies
- Load balancer config

**Phase 3: Parallel Testing** (3_parallel/)
Launch ${MAX_PARALLELISM} test agents:
- Unit tests per module (parallel)
- Integration tests per service (parallel)
- API endpoint tests (parallel by route)
- Frontend component tests (parallel)
- Performance tests (parallel scenarios)
- Security tests (parallel vulnerabilities)
- Load tests (parallel user patterns)

**Phase 4: Parallel Integration & Optimization**
While maintaining module independence:
- Parallel API integration tests
- Parallel frontend-backend integration
- Parallel performance optimization
- Parallel security hardening
- Parallel documentation generation

**Phase 5: Final Sequential Integration** (4_sequential/)
Only truly sequential tasks:
- System-wide integration test
- End-to-end user flow validation
- Production deployment preparation
- Final documentation review

### Parallel Execution Rules:
1. **Always seek parallelism**: Before any task, ask "what can run in parallel?"
2. **Dependency mapping**: Clearly define what each parallel task produces
3. **Integration points**: Define how parallel results merge
4. **Resource management**: Balance load across ${MAX_PARALLELISM} agents
5. **Continuous integration**: Merge parallel work frequently

### Test-Driven Parallel Development:
For each parallel task:
1. Write tests first (can be done in parallel)
2. Implement to pass tests (parallel execution)
3. Refactor in parallel (different aspects)
4. Integrate results (minimal sequential work)

### Success Metrics:
- Parallel execution efficiency: >80% of time in parallel tasks
- Test coverage: ${TEST_COVERAGE_TARGET}% across all modules
- Integration success rate: 100% on first attempt
- Performance: Meets all SLAs with parallel architecture

Display '<PARALLEL-IMPLEMENTATION-COMPLETE>' when finished.
IMPLEMENTATION_PHASE
}

# Generate quality standards
generate_quality_standards() {
    cat << 'QUALITY_STANDARDS'
## QUALITY STANDARDS & METHODOLOGY

### Code Quality:
- **Modularity**: All files ≤ 500 lines, functions ≤ 50 lines
- **Security**: No hardcoded secrets, comprehensive input validation
- **Testing**: ${TEST_COVERAGE_TARGET}% coverage with TDD London School
- **Documentation**: Self-documenting code with strategic comments
- **Performance**: Optimized critical paths with benchmarking

### Tool Utilization:
- **Research**: WebFetchTool for comprehensive knowledge gathering
- **Development**: Edit/Replace for code implementation
- **Analysis**: GlobTool/GrepTool for pattern detection
- **Execution**: Bash for system operations and testing
- **Parallel**: BatchTool for concurrent operations
- **Version Control**: Git with semantic commits

### Deliverables:
$(if [[ "$ARCHITECTURE_DIAGRAMS" == true ]]; then echo "- ARCHITECTURE.md with detailed Mermaid diagrams"; fi)
$(if [[ "$KEY_MODULES_ANALYSIS" == true ]]; then echo "- KEY_MODULES.md with critical success factors"; fi)
$(if [[ "$RISK_ANALYSIS" == true ]]; then echo "- RISK_FACTORS.md with mitigation strategies"; fi)
- Functionally-Complete.md tracking implementation progress
- Comprehensive test suites with ${TEST_COVERAGE_TARGET}% coverage
- User-friendly examples in examples/ directory
- Complete documentation in implementation/ directory

### Success Metrics:
- All quality gates passed
- Performance benchmarks met
- Security audit passed
- Documentation complete
- User acceptance achieved

Continue until all success criteria are met.
Display '<WORKFLOW-COMPLETE>' when the entire process is finished.
QUALITY_STANDARDS
}

# Generate parallel execution principles
generate_parallel_principles() {
    cat << 'PARALLEL_PRINCIPLES'
## PARALLEL-FIRST EXECUTION PRINCIPLES

### Core Mandate: MAXIMIZE PARALLELISM AT EVERY STEP

**The Parallel Mindset**:
Before EVERY task, ask yourself:
1. What parts of this can run in parallel?
2. How can I decompose this for parallel execution?
3. What are the true dependencies vs artificial sequencing?

### Parallel Execution Patterns:

**1. Decomposition Pattern** (for complex tasks):
```
Complex Task → Analyze → Decompose → Parallel Execute → Integrate
Example: Building a web app
- Parallel: Frontend, Backend, Database, DevOps setup
- Sequential: Only final integration
```

**2. Fan-Out Pattern** (for similar operations):
```
Task Type → Identify Instances → Fan Out (${MAX_PARALLELISM} agents) → Gather Results
Example: Creating API endpoints
- Parallel: User endpoints, Admin endpoints, Public endpoints
- Sequential: API gateway configuration
```

**3. Pipeline Pattern** (for staged work):
```
Stage 1 (parallel) → Stage 2 (parallel) → Stage 3 (parallel)
Example: Module development
- Stage 1: Parallel spec writing
- Stage 2: Parallel implementation
- Stage 3: Parallel testing
```

**4. Recursive Parallelism** (for hierarchical tasks):
```
Top Level → Parallel Decompose → Each Part Parallel Decompose → Recurse
Example: Documentation generation
- Parallel: Each directory gets an agent
- Recursive: Each agent parallelizes subdirectories
```

### Practical Examples:

**Writing a Complex Repository**:
1. **Initial Parallel Analysis**:
   - Agent 1: Analyze frontend requirements
   - Agent 2: Analyze backend requirements
   - Agent 3: Analyze data requirements
   - Agent 4: Analyze infrastructure needs
   - Agent 5: Analyze security requirements

2. **Parallel Module Creation**:
   - Agents 1-3: Frontend modules (auth, dashboard, settings)
   - Agents 4-6: Backend services (api, auth, business)
   - Agents 7-8: Database layer (models, migrations)
   - Agents 9-10: DevOps (Docker, CI/CD)

3. **Parallel Testing**:
   - Each module tested by dedicated agent
   - All tests run simultaneously
   - Integration tests parallelized by flow

4. **Parallel Integration**:
   - Frontend-backend integration (parallel by feature)
   - Service-to-service integration (parallel by interaction)
   - Only final e2e test is sequential

### Anti-Patterns to AVOID:

❌ **Sequential Thinking**:
```
DON'T: Step 1 → Step 2 → Step 3 → Step 4
DO: Steps 1,2,3 (parallel) → Step 4 (integration)
```

❌ **False Dependencies**:
```
DON'T: "Backend must be done before frontend"
DO: Develop with mocked interfaces in parallel
```

❌ **Monolithic Tasks**:
```
DON'T: "Build the entire auth system"
DO: Parallel tasks for JWT, OAuth, sessions, permissions
```

### Parallel Task Coordination:

**Clear Output Contracts**:
Every parallel task must define:
- What it produces (files, configs, artifacts)
- Where it places outputs
- What format outputs use
- How outputs integrate with others

**Integration Checkpoints**:
- After each parallel phase, minimal integration
- Validate interfaces match
- Resolve any conflicts
- Continue to next parallel phase

### Resource Optimization:
- Use up to ${MAX_PARALLELISM} agents concurrently
- Balance task complexity across agents
- Monitor and adjust parallelism based on performance
- Never leave agents idle - find more parallel work

**Remember**: The goal is >80% parallel execution time. Sequential work should be the exception, not the rule.

PARALLEL_PRINCIPLES
}

# Generate hierarchical documentation update phase
generate_documentation_update_phase() {
    cat << 'DOC_UPDATE_PHASE'
## CONTINUOUS DOCUMENTATION MAINTENANCE

### Hierarchical CLAUDE.md Update System:
Maintain repository-wide contextual memory through automated documentation updates.

**Trigger**: Every ${DOCUMENTATION_UPDATE_THRESHOLD} lines of code changes
**Parallelism**: Up to ${DOCUMENTATION_MAX_PARALLELISM} concurrent agents

### Graph-Based Update Process:

1. **Bottom-Up Traversal**:
   - Identify leaf directories (no subdirectories)
   - Queue directories with analyzed children
   - Process up to ${DOCUMENTATION_MAX_PARALLELISM} nodes concurrently

2. **Parallel Agent Tasks**:
   Each agent analyzes one directory node:
   - Scan all source files in directory
   - Synthesize child CLAUDE.md files
   - Extract specifications, architecture, key modules
   - Identify risk factors and dependencies
   - Generate/update local CLAUDE.md

3. **Hierarchical Synthesis**:
   - Parent directories incorporate child summaries
   - Propagate insights upward through tree
   - Root CLAUDE.md provides complete overview
   - Maintain consistency across all levels

### CLAUDE.md Structure:
- **Overview**: Module purpose and scope
- **Architecture**: Design patterns and decisions
- **Key Components**: Critical modules and interactions
- **Dependencies**: Internal/external dependencies
- **Risk Factors**: Issues and technical debt
- **Integration**: Connection points with other modules
- **Recent Changes**: Latest modifications summary

### Agent Task Template:
"Analyze directory: [path]
1. Examine all source files
2. Synthesize child CLAUDE.md files
3. Extract: specifications, architecture, key modules, risks, dependencies
4. Generate comprehensive CLAUDE.md following project standards"

**Auto-commit**: 'docs: hierarchical CLAUDE.md update - synchronized repository context'
DOC_UPDATE_PHASE
}

# Add documentation update configuration
DOCUMENTATION_UPDATE_ENABLED=true
DOCUMENTATION_UPDATE_THRESHOLD=100  # lines of code change
DOCUMENTATION_MAX_PARALLELISM=10

# Execute workflow
execute_workflow() {
    local allowed_tools=$(build_allowed_tools)
    local claude_flags=$(build_claude_flags)
    local workflow_prompt=$(generate_workflow_prompt)
    
    # Add special handling for agent-integration mode
    if [[ "$DEVELOPMENT_MODE" == "agent-integration" ]]; then
        workflow_prompt+="\n\n## AGENT INTEGRATION SPECIFICS:\n"
        workflow_prompt+="- Review existing implementation in ${AGENT_SDK_PATH}\n"
        workflow_prompt+="- Design integration maintaining existing API contracts\n"
        workflow_prompt+="- Consider MCP support and multi-agent workflows\n"
        workflow_prompt+="- Handle file/folder input mapping for UI uploads\n"
        workflow_prompt+="- Implement proper error handling and output management\n"
    fi
    
    # Execute Claude with the generated prompt
    claude "$workflow_prompt" \
        --allowedTools "$allowed_tools" \
        $claude_flags
}

# Main execution
main() {
    parse_args "$@"
    validate_config
    show_config
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN - Would execute the following:"
        echo "Project: $PROJECT_NAME"
        echo "README: $README_PATH"
        echo "Workflow: $WORKFLOW_MODE"
        echo "Mode: $DEVELOPMENT_MODE"
        echo "Allowed Tools: $(build_allowed_tools)"
        echo "Claude Flags: $(build_claude_flags)"
        echo ""
        echo "Would generate workflow prompt and execute Claude..."
        exit 0
    fi
    
    # Execute the workflow
    execute_workflow
}

# Execute main function with all arguments
main "$@"