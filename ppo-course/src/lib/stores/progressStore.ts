import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type ConceptId = string
export type MasteryLevel = 0 | 1 | 2 | 3 | 4 | 5

export interface Achievement {
  id: string
  name: string
  description: string
  unlockedAt?: Date
  icon: string
}

export interface LearningPath {
  currentChapter: number
  currentSection: string
  suggestedNext: string[]
}

interface ProgressState {
  currentChapter: number
  completedSections: Set<string>
  conceptMastery: Map<ConceptId, MasteryLevel>
  achievements: Achievement[]
  learningPath: LearningPath | null
  totalTimeSpent: number
  lastActiveDate: Date
}

interface ProgressActions {
  completeSection: (sectionId: string) => void
  updateConceptMastery: (conceptId: ConceptId, level: MasteryLevel) => void
  unlockAchievement: (achievementId: string) => void
  updateLearningPath: (path: LearningPath) => void
  incrementTimeSpent: (minutes: number) => void
  setCurrentChapter: (chapter: number) => void
}

export const useProgressStore = create<ProgressState & ProgressActions>()(
  persist(
    (set, get) => ({
      // Initial state
      currentChapter: 1,
      completedSections: new Set(),
      conceptMastery: new Map(),
      achievements: [],
      learningPath: null,
      totalTimeSpent: 0,
      lastActiveDate: new Date(),

      // Actions
      completeSection: (sectionId: string) => {
        set((state) => ({
          completedSections: new Set([...state.completedSections, sectionId]),
          lastActiveDate: new Date()
        }))
      },

      updateConceptMastery: (conceptId: ConceptId, level: MasteryLevel) => {
        set((state) => {
          const newMastery = new Map(state.conceptMastery)
          newMastery.set(conceptId, level)
          return { conceptMastery: newMastery }
        })
      },

      unlockAchievement: (achievementId: string) => {
        const achievement = get().achievements.find(a => a.id === achievementId)
        if (achievement && !achievement.unlockedAt) {
          set((state) => ({
            achievements: state.achievements.map(a =>
              a.id === achievementId
                ? { ...a, unlockedAt: new Date() }
                : a
            )
          }))
        }
      },

      updateLearningPath: (path: LearningPath) => {
        set({ learningPath: path })
      },

      incrementTimeSpent: (minutes: number) => {
        set((state) => ({
          totalTimeSpent: state.totalTimeSpent + minutes,
          lastActiveDate: new Date()
        }))
      },

      setCurrentChapter: (chapter: number) => {
        set({ currentChapter: chapter })
      }
    }),
    {
      name: 'ppo-course-progress',
      // Custom serialization for Set and Map
      storage: {
        getItem: (name) => {
          const str = localStorage.getItem(name)
          if (!str) return null
          
          const state = JSON.parse(str)
          return {
            ...state,
            state: {
              ...state.state,
              completedSections: new Set(state.state.completedSections || []),
              conceptMastery: new Map(state.state.conceptMastery || []),
              lastActiveDate: new Date(state.state.lastActiveDate)
            }
          }
        },
        setItem: (name, value) => {
          const state = {
            ...value,
            state: {
              ...value.state,
              completedSections: Array.from(value.state.completedSections),
              conceptMastery: Array.from(value.state.conceptMastery.entries())
            }
          }
          localStorage.setItem(name, JSON.stringify(state))
        },
        removeItem: (name) => localStorage.removeItem(name)
      }
    }
  )
)