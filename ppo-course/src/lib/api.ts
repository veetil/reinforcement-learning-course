const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface TrainingConfig {
  environment: string;
  algorithm: string;
  hyperparameters: {
    learningRate: number;
    clipRange: number;
    gamma: number;
    gaeBalance: number;
    nEpochs: number;
    batchSize: number;
    entropyCoef: number;
    valueCoef: number;
  };
}

export interface TrainingStatus {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  currentEpisode: number;
  totalEpisodes: number;
  metrics: {
    meanReward: number;
    meanEpisodeLength: number;
    policyLoss: number;
    valueLoss: number;
    entropy: number;
    klDivergence: number;
    clipFraction: number;
  };
}

export interface CodeExecutionRequest {
  code: string;
  language: 'python' | 'javascript';
  timeout?: number;
}

export interface CodeExecutionResult {
  output: string;
  error?: string;
  executionTime: number;
}

class APIClient {
  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    return response.json();
  }

  async startTraining(config: TrainingConfig): Promise<{ id: string }> {
    return this.fetch('/api/v1/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async getTrainingStatus(id: string): Promise<TrainingStatus> {
    return this.fetch(`/api/v1/training/status/${id}`);
  }

  async stopTraining(id: string): Promise<void> {
    await this.fetch(`/api/v1/training/stop/${id}`, {
      method: 'POST',
    });
  }

  async executeCode(request: CodeExecutionRequest): Promise<CodeExecutionResult> {
    return this.fetch('/api/v1/code/execute', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getEnvironments(): Promise<string[]> {
    return this.fetch('/api/v1/training/environments');
  }

  async getAlgorithms(): Promise<string[]> {
    return this.fetch('/api/v1/training/algorithms');
  }

  async saveCheckpoint(trainingId: string, name: string): Promise<void> {
    await this.fetch('/api/v1/training/checkpoint', {
      method: 'POST',
      body: JSON.stringify({ trainingId, name }),
    });
  }

  async loadCheckpoint(checkpointId: string): Promise<TrainingConfig> {
    return this.fetch(`/api/v1/training/checkpoint/${checkpointId}`);
  }

  async exportModel(trainingId: string, format: 'onnx' | 'pytorch' | 'tensorflow'): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/api/v1/training/export/${trainingId}?format=${format}`);
    
    if (!response.ok) {
      throw new Error(`Export failed: ${response.statusText}`);
    }

    return response.blob();
  }

  // WebSocket connection for real-time updates
  connectToTraining(trainingId: string, onUpdate: (status: TrainingStatus) => void): () => void {
    const ws = new WebSocket(`${API_BASE_URL.replace('http', 'ws')}/ws/training/${trainingId}`);
    
    ws.onmessage = (event) => {
      const status = JSON.parse(event.data) as TrainingStatus;
      onUpdate(status);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    // Return cleanup function
    return () => {
      ws.close();
    };
  }
}

export const api = new APIClient();