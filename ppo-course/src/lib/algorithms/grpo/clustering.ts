export class KMeans {
  private centroids: number[][] = [];
  private k: number;

  constructor(k: number) {
    this.k = k;
  }

  async fit(data: number[][]): Promise<number[]> {
    if (data.length === 0) {
      return [];
    }

    // Initialize centroids randomly
    this.centroids = this.initializeCentroids(data);

    // Iterate until convergence
    let clusters: number[] = [];
    let previousClusters: number[] = [];
    let iterations = 0;
    const maxIterations = 100;

    do {
      previousClusters = [...clusters];
      
      // Assign points to nearest centroid
      clusters = data.map(point => this.assignToCluster(point));
      
      // Update centroids
      this.updateCentroids(data, clusters);
      
      iterations++;
    } while (!this.hasConverged(clusters, previousClusters) && iterations < maxIterations);

    return clusters;
  }

  private initializeCentroids(data: number[][]): number[][] {
    // K-means++ initialization
    const centroids: number[][] = [];
    const indices = new Set<number>();

    // First centroid is random
    const firstIndex = Math.floor(Math.random() * data.length);
    centroids.push([...data[firstIndex]]);
    indices.add(firstIndex);

    // Select remaining centroids
    for (let i = 1; i < this.k; i++) {
      const distances = data.map((point, idx) => {
        if (indices.has(idx)) return 0;
        
        // Find minimum distance to existing centroids
        const minDist = Math.min(...centroids.map(c => this.euclideanDistance(point, c)));
        return minDist * minDist; // Square for probability weighting
      });

      // Select point with probability proportional to squared distance
      const totalDist = distances.reduce((a, b) => a + b, 0);
      let random = Math.random() * totalDist;
      
      for (let j = 0; j < data.length; j++) {
        random -= distances[j];
        if (random <= 0 && !indices.has(j)) {
          centroids.push([...data[j]]);
          indices.add(j);
          break;
        }
      }
    }

    return centroids;
  }

  private assignToCluster(point: number[]): number {
    let minDistance = Infinity;
    let closestCluster = 0;

    for (let i = 0; i < this.centroids.length; i++) {
      const distance = this.euclideanDistance(point, this.centroids[i]);
      if (distance < minDistance) {
        minDistance = distance;
        closestCluster = i;
      }
    }

    return closestCluster;
  }

  private updateCentroids(data: number[][], clusters: number[]): void {
    const newCentroids: number[][] = Array(this.k).fill(null).map(() => []);
    const counts: number[] = Array(this.k).fill(0);

    // Sum points in each cluster
    for (let i = 0; i < data.length; i++) {
      const cluster = clusters[i];
      if (newCentroids[cluster].length === 0) {
        newCentroids[cluster] = Array(data[i].length).fill(0);
      }
      
      for (let j = 0; j < data[i].length; j++) {
        newCentroids[cluster][j] += data[i][j];
      }
      counts[cluster]++;
    }

    // Average to get new centroids
    for (let i = 0; i < this.k; i++) {
      if (counts[i] > 0) {
        this.centroids[i] = newCentroids[i].map(sum => sum / counts[i]);
      }
    }
  }

  private euclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
  }

  private hasConverged(clusters: number[], previousClusters: number[]): boolean {
    if (clusters.length !== previousClusters.length) return false;
    
    for (let i = 0; i < clusters.length; i++) {
      if (clusters[i] !== previousClusters[i]) return false;
    }
    
    return true;
  }
}