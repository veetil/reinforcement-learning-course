// Learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom'

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock framer-motion
jest.mock('framer-motion', () => {
  const React = require('react');
  
  // Create a generic motion component factory
  const createMotionComponent = (element) => {
    return React.forwardRef(({ children, whileHover, whileTap, layoutId, initial, animate, exit, transition, ...props }, ref) => {
      return React.createElement(element, { ...props, ref }, children);
    });
  };
  
  return {
    motion: {
      div: createMotionComponent('div'),
      circle: createMotionComponent('circle'),
      button: createMotionComponent('button'),
      span: createMotionComponent('span'),
      svg: createMotionComponent('svg'),
      path: createMotionComponent('path'),
      rect: createMotionComponent('rect'),
      g: createMotionComponent('g'),
    },
    AnimatePresence: ({ children }) => children,
  };
});