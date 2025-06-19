import { useState, useEffect, useCallback } from 'react';
import { api, TrainingConfig, TrainingStatus } from '@/lib/api';

export function useTraining() {
  const [trainingId, setTrainingId] = useState<string | null>(null);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Start new training
  const startTraining = useCallback(async (config: TrainingConfig) => {
    try {
      setError(null);
      const { id } = await api.startTraining(config);
      setTrainingId(id);
      return id;
    } catch (err) {
      setError((err as Error).message);
      throw err;
    }
  }, []);

  // Stop current training
  const stopTraining = useCallback(async () => {
    if (!trainingId) return;
    
    try {
      await api.stopTraining(trainingId);
      setTrainingId(null);
      setStatus(null);
    } catch (err) {
      setError((err as Error).message);
    }
  }, [trainingId]);

  // Connect to training updates
  useEffect(() => {
    if (!trainingId) {
      setIsConnected(false);
      return;
    }

    let cleanup: (() => void) | null = null;

    const connect = async () => {
      try {
        // Get initial status
        const initialStatus = await api.getTrainingStatus(trainingId);
        setStatus(initialStatus);
        
        // Connect to WebSocket for updates
        cleanup = api.connectToTraining(trainingId, (newStatus) => {
          setStatus(newStatus);
          setIsConnected(true);
        });
      } catch (err) {
        setError((err as Error).message);
        setIsConnected(false);
      }
    };

    connect();

    return () => {
      cleanup?.();
      setIsConnected(false);
    };
  }, [trainingId]);

  return {
    trainingId,
    status,
    isConnected,
    error,
    startTraining,
    stopTraining,
  };
}

// Hook for code execution
export function useCodeExecution() {
  const [isExecuting, setIsExecuting] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const executeCode = useCallback(async (code: string, language: 'python' | 'javascript' = 'python') => {
    setIsExecuting(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.executeCode({ code, language, timeout: 30000 });
      
      if (response.error) {
        setError(response.error);
      } else {
        setResult(response.output);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsExecuting(false);
    }
  }, []);

  return {
    executeCode,
    isExecuting,
    result,
    error,
  };
}