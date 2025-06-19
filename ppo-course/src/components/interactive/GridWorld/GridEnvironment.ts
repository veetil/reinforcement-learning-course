export enum Action {
  UP = 'UP',
  DOWN = 'DOWN',
  LEFT = 'LEFT',
  RIGHT = 'RIGHT',
}

export interface Position {
  x: number;
  y: number;
}

export interface GridState {
  agentPosition: Position;
  goalPosition: Position;
  obstacles: Position[];
  totalReward: number;
  stepCount: number;
  isDone: boolean;
  trajectory: Position[];
}

export interface StepResult {
  state: GridState;
  reward: number;
  done: boolean;
  info: {
    action: Action;
    collision?: boolean;
    reachedGoal?: boolean;
  };
}

export class GridEnvironment {
  public readonly width: number;
  public readonly height: number;
  
  private state: GridState;
  private readonly stepPenalty = -1;
  private readonly goalReward = 10;
  private readonly collisionPenalty = -5;
  
  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.state = this.createInitialState();
  }
  
  private createInitialState(): GridState {
    return {
      agentPosition: { x: 0, y: 0 },
      goalPosition: { x: this.width - 1, y: this.height - 1 },
      obstacles: [],
      totalReward: 0,
      stepCount: 0,
      isDone: false,
      trajectory: [{ x: 0, y: 0 }],
    };
  }
  
  public reset(): GridState {
    this.state = this.createInitialState();
    return this.getState();
  }
  
  public getState(): GridState {
    return { ...this.state };
  }
  
  public addObstacle(x: number, y: number): void {
    if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
      const isOccupied = this.state.obstacles.some(
        obs => obs.x === x && obs.y === y
      );
      if (!isOccupied && 
          !(x === this.state.agentPosition.x && y === this.state.agentPosition.y) &&
          !(x === this.state.goalPosition.x && y === this.state.goalPosition.y)) {
        this.state.obstacles.push({ x, y });
      }
    }
  }
  
  public removeObstacle(x: number, y: number): void {
    this.state.obstacles = this.state.obstacles.filter(
      obs => !(obs.x === x && obs.y === y)
    );
  }
  
  public step(action: Action): StepResult {
    if (this.state.isDone) {
      return {
        state: this.getState(),
        reward: 0,
        done: true,
        info: { action },
      };
    }
    
    // Calculate new position
    const newPosition = this.getNewPosition(action);
    
    // Check if move is valid
    const isValidMove = this.isValidPosition(newPosition);
    const hasCollision = this.hasObstacleAt(newPosition);
    
    let reward = this.stepPenalty;
    const info: StepResult['info'] = { action };
    
    if (!isValidMove || hasCollision) {
      // Invalid move or collision
      if (hasCollision) {
        reward = this.collisionPenalty;
        info.collision = true;
      }
    } else {
      // Valid move
      this.state.agentPosition = newPosition;
      this.state.trajectory.push({ ...newPosition });
      
      // Check if reached goal
      if (this.isAtGoal()) {
        reward = this.goalReward;
        this.state.isDone = true;
        info.reachedGoal = true;
      }
    }
    
    // Update state
    this.state.totalReward += reward;
    this.state.stepCount += 1;
    
    return {
      state: this.getState(),
      reward,
      done: this.state.isDone,
      info,
    };
  }
  
  private getNewPosition(action: Action): Position {
    const { x, y } = this.state.agentPosition;
    
    switch (action) {
      case Action.UP:
        return { x, y: y - 1 };
      case Action.DOWN:
        return { x, y: y + 1 };
      case Action.LEFT:
        return { x: x - 1, y };
      case Action.RIGHT:
        return { x: x + 1, y };
    }
  }
  
  private isValidPosition(pos: Position): boolean {
    return pos.x >= 0 && pos.x < this.width && 
           pos.y >= 0 && pos.y < this.height;
  }
  
  private hasObstacleAt(pos: Position): boolean {
    return this.state.obstacles.some(
      obs => obs.x === pos.x && obs.y === pos.y
    );
  }
  
  private isAtGoal(): boolean {
    return this.state.agentPosition.x === this.state.goalPosition.x &&
           this.state.agentPosition.y === this.state.goalPosition.y;
  }
  
  public getValidActions(): Action[] {
    const actions: Action[] = [];
    
    for (const action of Object.values(Action)) {
      const newPos = this.getNewPosition(action);
      if (this.isValidPosition(newPos) && !this.hasObstacleAt(newPos)) {
        actions.push(action);
      }
    }
    
    return actions;
  }
  
  public getRewardAt(x: number, y: number): number {
    if (x === this.state.goalPosition.x && y === this.state.goalPosition.y) {
      return this.goalReward;
    }
    if (this.hasObstacleAt({ x, y })) {
      return this.collisionPenalty;
    }
    return this.stepPenalty;
  }
}