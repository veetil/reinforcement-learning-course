'use client';

import React from 'react';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { AlgorithmComparison } from '@/components/algorithms/AlgorithmComparison';

export default function ComparePage() {
  return (
    <div className="container mx-auto py-8 px-4 max-w-7xl">
      <div className="mb-8">
        <Link href="/algorithms">
          <Button variant="ghost" className="mb-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Algorithm Zoo
          </Button>
        </Link>
        
        <h1 className="text-4xl font-bold mb-2">Algorithm Comparison Tool</h1>
        <p className="text-xl text-muted-foreground">
          Compare different RL algorithms side-by-side across various environments and metrics
        </p>
      </div>

      <AlgorithmComparison />
    </div>
  );
}