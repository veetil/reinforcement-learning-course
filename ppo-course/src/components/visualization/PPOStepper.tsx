'use client'

import React, { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, SkipForward, RotateCcw, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

interface PPOStep {
  id: string
  name: string
  description: string
  duration: number
  visualization?: React.ReactNode
  data?: any
}

interface PPOStepperProps {
  steps: PPOStep[]
  onStepComplete?: (step: PPOStep, data: any) => void
  autoPlay?: boolean
  className?: string
}

export default function PPOStepper({
  steps,
  onStepComplete,
  autoPlay = false,
  className
}: PPOStepperProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(autoPlay)
  const [stepData, setStepData] = useState<Record<string, any>>({})
  const [isAnimating, setIsAnimating] = useState(false)

  const handleNextStep = useCallback(() => {
    if (currentStep < steps.length - 1) {
      setIsAnimating(true)
      const step = steps[currentStep]
      
      // Store step data
      const newData = { ...stepData, [step.id]: step.data }
      setStepData(newData)
      
      // Notify parent
      onStepComplete?.(step, step.data)
      
      setTimeout(() => {
        setCurrentStep(currentStep + 1)
        setIsAnimating(false)
      }, 300)
    } else if (currentStep === steps.length - 1) {
      setIsPlaying(false)
    }
  }, [currentStep, steps, stepData, onStepComplete])

  const handlePreviousStep = useCallback(() => {
    if (currentStep > 0) {
      setIsAnimating(true)
      setTimeout(() => {
        setCurrentStep(currentStep - 1)
        setIsAnimating(false)
      }, 300)
    }
  }, [currentStep])

  const handleReset = useCallback(() => {
    setCurrentStep(0)
    setStepData({})
    setIsPlaying(false)
    setIsAnimating(false)
  }, [])

  const togglePlay = useCallback(() => {
    setIsPlaying(!isPlaying)
  }, [isPlaying])

  // Auto-advance when playing
  React.useEffect(() => {
    if (isPlaying && !isAnimating) {
      const timer = setTimeout(() => {
        handleNextStep()
      }, steps[currentStep]?.duration || 2000)
      
      return () => clearTimeout(timer)
    }
  }, [isPlaying, currentStep, isAnimating, handleNextStep, steps])

  const currentStepData = steps[currentStep]

  return (
    <div className={cn("bg-white rounded-lg shadow-lg p-6", className)}>
      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold">PPO Training Steps</h3>
          <span className="text-sm text-gray-500">
            Step {currentStep + 1} of {steps.length}
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <motion.div
            className="bg-blue-600 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      {/* Step Indicators */}
      <div className="flex items-center justify-between mb-6 overflow-x-auto">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={cn(
              "flex items-center",
              index < steps.length - 1 && "flex-1"
            )}
          >
            <button
              onClick={() => setCurrentStep(index)}
              disabled={isPlaying}
              className={cn(
                "w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium transition-all",
                index === currentStep
                  ? "bg-blue-600 text-white scale-110 shadow-lg"
                  : index < currentStep
                  ? "bg-green-500 text-white"
                  : "bg-gray-200 text-gray-600"
              )}
            >
              {index + 1}
            </button>
            {index < steps.length - 1 && (
              <ChevronRight
                className={cn(
                  "mx-2 flex-shrink-0",
                  index < currentStep ? "text-green-500" : "text-gray-300"
                )}
                size={20}
              />
            )}
          </div>
        ))}
      </div>

      {/* Current Step Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="mb-6"
        >
          <h4 className="text-xl font-semibold mb-2">{currentStepData.name}</h4>
          <p className="text-gray-600 mb-4">{currentStepData.description}</p>
          
          {/* Visualization Area */}
          {currentStepData.visualization && (
            <div className="bg-gray-50 rounded-lg p-4 min-h-[300px] flex items-center justify-center">
              {currentStepData.visualization}
            </div>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Control Buttons */}
      <div className="flex items-center justify-center space-x-4">
        <button
          onClick={handlePreviousStep}
          disabled={currentStep === 0 || isPlaying}
          className={cn(
            "px-4 py-2 rounded-lg flex items-center space-x-2 transition-all",
            currentStep === 0 || isPlaying
              ? "bg-gray-100 text-gray-400 cursor-not-allowed"
              : "bg-gray-200 text-gray-700 hover:bg-gray-300"
          )}
        >
          <SkipForward className="rotate-180" size={20} />
          <span>Previous</span>
        </button>

        <button
          onClick={togglePlay}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg flex items-center space-x-2 hover:bg-blue-700 transition-all shadow-lg"
        >
          {isPlaying ? (
            <>
              <Pause size={20} />
              <span>Pause</span>
            </>
          ) : (
            <>
              <Play size={20} />
              <span>Play</span>
            </>
          )}
        </button>

        <button
          onClick={handleNextStep}
          disabled={currentStep === steps.length - 1 || isPlaying}
          className={cn(
            "px-4 py-2 rounded-lg flex items-center space-x-2 transition-all",
            currentStep === steps.length - 1 || isPlaying
              ? "bg-gray-100 text-gray-400 cursor-not-allowed"
              : "bg-gray-200 text-gray-700 hover:bg-gray-300"
          )}
        >
          <span>Next</span>
          <SkipForward size={20} />
        </button>

        <button
          onClick={handleReset}
          className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg flex items-center space-x-2 hover:bg-gray-300 transition-all"
        >
          <RotateCcw size={20} />
          <span>Reset</span>
        </button>
      </div>

      {/* Step Data Display (for debugging/learning) */}
      {Object.keys(stepData).length > 0 && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="mt-6 p-4 bg-gray-100 rounded-lg"
        >
          <h5 className="font-semibold mb-2">Collected Data:</h5>
          <pre className="text-xs overflow-x-auto">
            {JSON.stringify(stepData, null, 2)}
          </pre>
        </motion.div>
      )}
    </div>
  )
}