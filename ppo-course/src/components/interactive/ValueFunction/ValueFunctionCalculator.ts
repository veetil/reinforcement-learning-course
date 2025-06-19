export class ValueFunctionCalculator {
  public width: number;
  public height: number;
  public values: number[][];
  public goalX: number = -1;
  public goalY: number = -1;
  private obstacles: Set<string> = new Set();
  private discountFactor: number = 0.9;
  private convergenceThreshold: number = 0.001;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.values = Array(height).fill(null).map(() => Array(width).fill(0));
  }

  setGoal(x: number, y: number): void {
    this.goalX = x;
    this.goalY = y;
  }

  addObstacle(x: number, y: number): void {
    this.obstacles.add(`${x},${y}`);
  }

  removeObstacle(x: number, y: number): void {
    this.obstacles.delete(`${x},${y}`);
  }

  setDiscountFactor(gamma: number): void {
    this.discountFactor = Math.max(0, Math.min(1, gamma));
  }

  getValue(x: number, y: number): number {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      return 0;
    }
    return this.values[y][x];
  }

  isObstacle(x: number, y: number): boolean {
    return this.obstacles.has(`${x},${y}`);
  }

  isGoal(x: number, y: number): boolean {
    return x === this.goalX && y === this.goalY;
  }

  calculateValues(): number {
    if (this.goalX === -1 || this.goalY === -1) {
      return 0;
    }

    // Initialize values
    const newValues = Array(this.height).fill(null).map(() => Array(this.width).fill(0));
    
    // Set goal value
    newValues[this.goalY][this.goalX] = 1.0;

    let iterations = 0;
    let maxDelta = Infinity;

    // Value iteration algorithm
    while (maxDelta > this.convergenceThreshold && iterations < 100) {
      maxDelta = 0;
      const tempValues = newValues.map(row => [...row]);

      for (let y = 0; y < this.height; y++) {
        for (let x = 0; x < this.width; x++) {
          if (this.isObstacle(x, y)) {
            tempValues[y][x] = 0;
            continue;
          }

          if (this.isGoal(x, y)) {
            tempValues[y][x] = 1.0;
            continue;
          }

          // Calculate max value from neighboring states
          const neighbors = [
            { x: x + 1, y },
            { x: x - 1, y },
            { x, y: y + 1 },
            { x, y: y - 1 }
          ];

          let maxNeighborValue = 0;
          for (const neighbor of neighbors) {
            if (neighbor.x >= 0 && neighbor.x < this.width &&
                neighbor.y >= 0 && neighbor.y < this.height &&
                !this.isObstacle(neighbor.x, neighbor.y)) {
              maxNeighborValue = Math.max(maxNeighborValue, newValues[neighbor.y][neighbor.x]);
            }
          }

          const oldValue = newValues[y][x];
          const newValue = this.discountFactor * maxNeighborValue;
          tempValues[y][x] = newValue;
          maxDelta = Math.max(maxDelta, Math.abs(newValue - oldValue));
        }
      }

      // Update values
      for (let y = 0; y < this.height; y++) {
        for (let x = 0; x < this.width; x++) {
          newValues[y][x] = tempValues[y][x];
        }
      }

      iterations++;
    }

    this.values = newValues;
    return iterations;
  }

  getOptimalPolicy(): { [key: string]: string } {
    const policy: { [key: string]: string } = {};

    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        if (this.isObstacle(x, y) || this.isGoal(x, y)) {
          continue;
        }

        const actions = [
          { dx: 1, dy: 0, name: 'right' },
          { dx: -1, dy: 0, name: 'left' },
          { dx: 0, dy: 1, name: 'down' },
          { dx: 0, dy: -1, name: 'up' }
        ];

        let bestAction = 'none';
        let bestValue = -Infinity;

        for (const action of actions) {
          const nextX = x + action.dx;
          const nextY = y + action.dy;

          if (nextX >= 0 && nextX < this.width &&
              nextY >= 0 && nextY < this.height &&
              !this.isObstacle(nextX, nextY)) {
            const value = this.getValue(nextX, nextY);
            if (value > bestValue) {
              bestValue = value;
              bestAction = action.name;
            }
          }
        }

        policy[`${x},${y}`] = bestAction;
      }
    }

    return policy;
  }

  reset(): void {
    this.values = Array(this.height).fill(null).map(() => Array(this.width).fill(0));
    this.goalX = -1;
    this.goalY = -1;
    this.obstacles.clear();
  }

  exportData(): {
    width: number;
    height: number;
    values: number[][];
    goal: { x: number; y: number };
    obstacles: Array<{ x: number; y: number }>;
    discountFactor: number;
  } {
    return {
      width: this.width,
      height: this.height,
      values: this.values,
      goal: { x: this.goalX, y: this.goalY },
      obstacles: Array.from(this.obstacles).map(key => {
        const [x, y] = key.split(',').map(Number);
        return { x, y };
      }),
      discountFactor: this.discountFactor
    };
  }
}