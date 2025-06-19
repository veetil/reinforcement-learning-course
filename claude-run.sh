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
    README_PATH     Path to initial requirements/readme file (default: README.md)

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
    
    # Agent integration with iterative planning
    ./claude-run.sh --workflow iterative --mode agent-integration --enable-mcp agent-app readme2.md
    
    # Hybrid workflow with architecture analysis
    ./claude-run.sh --workflow hybrid --arch-diagrams --key-modules --risk-analysis my-project spec.md
    
    # Full development with GitHub integration
    ./claude-run.sh --github-repo myuser/myproject --iterations 5 full-app requirements.md

PHASES:
    1. Research & Discovery     - Comprehensive domain and technology research
    2. Specification           - Requirements analysis and system specification
    3. Architecture            - System design with diagrams and analysis
    4. Iterative Planning      - Multi-round planning with metrics evaluation
    5. Implementation          - TDD-based incremental development
    6. Integration & Testing   - Comprehensive testing and quality assurance
    7. Deployment             - Production readiness and deployment

For more information, see CLAUDE-RUN-GUIDE.md
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
                if [[ "$PROJECT_NAME" == "claude-project" ]]; then
                    PROJECT_NAME="$1"
                elif [[ "$README_PATH" == "README.md" ]]; then
                    README_PATH="$1"
                else
                    echo "Too many arguments: $1" >&2
                    echo "Use --help for usage information" >&2
                    exit 1
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
    
    # Check if README exists
    if [[ ! -f "$README_PATH" ]]; then
        # Try common README file variations
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
            echo "Please specify a valid README file path or create one of the above files." >&2
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
Commit Frequency: $COMMIT_FREQUENCY
GitHub Repo: ${GITHUB_REPO:-none}
Output Format: $OUTPUT_FORMAT
Metrics Enabled: $METRICS_ENABLED
Architecture Diagrams: $ARCHITECTURE_DIAGRAMS
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
    
    echo -e "$prompt"
}

# Generate research phase
generate_research_phase() {
    cat << 'RESEARCH_PHASE'
## PHASE 0: COMPREHENSIVE RESEARCH & DISCOVERY
### Research Depth: ${RESEARCH_DEPTH:-standard}
### Parallel Web Research Phase:

1. **Domain Research**:
   - WebFetchTool: Extract key concepts from ${README_PATH}
   - WebFetchTool: Search for latest industry trends and technologies
   - WebFetchTool: Research competitive landscape and existing solutions
   $(if [[ "${RESEARCH_DEPTH:-standard}" == "comprehensive" ]]; then echo "   - WebFetchTool: Gather academic papers and technical documentation"; fi)

2. **Technology Stack Research**:
   - WebFetchTool: Research best practices for identified technology domains
   - WebFetchTool: Search for framework comparisons and recommendations
   - WebFetchTool: Investigate security considerations and compliance requirements
   $(if [[ "${RESEARCH_DEPTH:-standard}" != "basic" ]]; then echo "   - WebFetchTool: Research scalability patterns and architecture approaches"; fi)

3. **Implementation Research**:
   - WebFetchTool: Search for code examples and implementation patterns
   - WebFetchTool: Research testing frameworks and methodologies
   - WebFetchTool: Investigate deployment and DevOps best practices
   $(if [[ "${RESEARCH_DEPTH:-standard}" == "comprehensive" ]]; then echo "   - WebFetchTool: Research monitoring and observability solutions"; fi)

$(if [[ "$DEVELOPMENT_MODE" == "agent-integration" ]]; then cat << 'AGENT_RESEARCH'
4. **Agent System Research**:
   - WebFetchTool: Research Claude CLI SDK architecture and capabilities
   - WebFetchTool: Investigate MCP integration patterns
   - WebFetchTool: Study agent orchestration and workflow management
   - Review ${AGENT_SDK_PATH} for existing implementation patterns
AGENT_RESEARCH
fi)

### Research Processing:
Use BatchTool to execute all research queries in parallel for maximum efficiency.

**Commit**: 'feat: complete comprehensive research phase - gathered domain knowledge, technology insights, and implementation patterns'
RESEARCH_PHASE
}

# Generate SPARC phases
generate_sparc_phases() {
    cat << 'SPARC_PHASES'
## SPECIFICATION PHASE
### Requirements Analysis:
1. **Functional Requirements**:
   - Analyze ${README_PATH} to extract core functionality
   - Define user stories and acceptance criteria
   - Identify system boundaries and interfaces
   - Specify API endpoints and data models
   - Define user interface requirements and user experience flows

2. **Non-Functional Requirements**:
   - Security and compliance requirements
   - Performance benchmarks and SLAs
   - Scalability and availability targets
   - Maintainability and extensibility goals

3. **Technical Constraints**:
   - Technology stack decisions based on research
   - Integration requirements and dependencies
   - Deployment and infrastructure constraints
   - Budget and timeline considerations

**Commit**: 'docs: complete specification phase - defined functional/non-functional requirements and technical constraints'

## PSEUDOCODE PHASE
### High-Level Architecture Design:
1. **System Architecture**:
   - Define components and their responsibilities
   - Design data flow and communication patterns
   - Specify APIs and integration points
   - Plan error handling and recovery strategies

2. **Algorithm Design**:
   - Core business logic algorithms
   - Data processing and transformation logic
   - Optimization strategies and performance considerations
   - Security and validation algorithms

3. **Test Strategy**:
   - Unit testing approach (TDD London School)
   - Integration testing strategy
   - End-to-end testing scenarios
   - Target: ${TEST_COVERAGE_TARGET}% test coverage

**Commit**: 'design: complete pseudocode phase - defined system architecture, algorithms, and test strategy'

## ARCHITECTURE PHASE
### Detailed System Design:
1. **Component Architecture**:
   - Detailed component specifications
   - Interface definitions and contracts
   - Dependency injection and inversion of control
   - Configuration management strategy

2. **Data Architecture**:
   - Database schema design
   - Data access patterns and repositories
   - Caching strategies and data flow
   - Backup and recovery procedures

3. **Infrastructure Architecture**:
   - Deployment architecture and environments
   - CI/CD pipeline design
   - Monitoring and logging architecture
   - Security architecture and access controls

$(if [[ "$ARCHITECTURE_DIAGRAMS" == true ]]; then echo "Generate comprehensive Mermaid diagrams in ARCHITECTURE.md"; fi)
$(if [[ "$KEY_MODULES_ANALYSIS" == true ]]; then echo "Create KEY_MODULES.md with critical success factors analysis"; fi)
$(if [[ "$RISK_ANALYSIS" == true ]]; then echo "Generate RISK_FACTORS.md with detailed risk assessment"; fi)

**Commit**: 'arch: complete architecture phase - detailed component, data, and infrastructure design'

## REFINEMENT PHASE (TDD Implementation)
### Development Tracks:
[Implementation details based on development mode]
SPARC_PHASES
}

# Generate iterative phases
generate_iterative_phases() {
    cat << 'ITERATIVE_PHASES'
## ITERATIVE PLANNING PHASE
### Initial Planning Round:
1. **Phase Analysis**:
   - Extract phases from ${README_PATH}
   - Create initial plan in plan/1.md, 2.md, 3.md
   - Define incremental steps with clear objectives
   - Establish testing criteria for each step

2. **Metrics Definition**:
   - Define SPCT (Speed, Performance, Correctness, Thoughtfulness) metrics
   - Establish evaluation criteria for plan quality
   - Create benchmarks for success measurement
   - Design feedback loops for continuous improvement

### Iterative Refinement (${ITERATION_COUNT} iterations):
For iteration in 1..${ITERATION_COUNT}:
   1. **Evaluation**:
      - Assess plan quality using SPCT metrics
      - Analyze alignment with critical success factors
      - Review feasibility and risk factors
      - Gather feedback from previous iterations
   
   2. **Deep Research**:
      - WebFetchTool: Research gaps identified in evaluation
      - Focus on critical modules and success factors
      - Investigate solutions to identified risks
      - Study best practices for weak areas
   
   3. **Plan Improvement**:
      - Refine steps based on evaluation results
      - Add detail to underspecified areas
      - Remove or rework problematic steps
      - Write improved plan to plan_r${iteration}/
   
   4. **Validation**:
      - Verify improved plan addresses identified issues
      - Check alignment with project objectives
      - Ensure maintainability and scalability
      - Confirm testability of all components

### Final Plan:
- Rename best iteration to plan_final/
- Generate comprehensive documentation
- Create execution roadmap with milestones
- Define success criteria and KPIs

**Commit**: 'plan: complete iterative planning with ${ITERATION_COUNT} refinement rounds'
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
## IMPLEMENTATION PHASE
### Progressive Development Process:

For each step in plan_final/:
   1. **Test-Driven Development**:
      - Write comprehensive test suite for the step
      - Implement functionality to pass tests
      - Refactor for quality and maintainability
      - Achieve ${TEST_COVERAGE_TARGET}% test coverage
   
   2. **Quality Assurance**:
      - Run all tests and fix any failures
      - Perform code review and optimization
      - Update documentation as needed
      - Validate against requirements
   
   3. **Examples & Documentation**:
      - Create user-friendly examples in examples/
      - Test all examples for correctness
      - Document implementation in implementation/${step}.md
      - Update Functionally-Complete.md with progress
   
   4. **Version Control**:
      $(if [[ -n "$GITHUB_REPO" ]]; then echo "- Push changes to GitHub repo: ${GITHUB_REPO}"; else echo "- Commit changes locally"; fi)
      - Use semantic commit messages
      - Tag important milestones
      - Maintain clean commit history

### Continuous Integration:
- Run tests on every commit
- Monitor code quality metrics
- Track progress against plan
- Update status documentation

### User Interaction Points:
- Maximum 2-3 stops for critical decisions
- Proactive clarification requests upfront
- Clear communication of blockers
- Efficient resolution of ambiguities

**Success Criteria**:
- All planned features implemented
- ${TEST_COVERAGE_TARGET}% test coverage achieved
- All examples working correctly
- Documentation complete and accurate
- Production-ready code delivered

Display '<IMPLEMENTATION-COMPLETE>' when finished.
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