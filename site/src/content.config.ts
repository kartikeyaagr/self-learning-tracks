import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

// One course = src/content/courses/<dir>/course.json + lessons/*.md (written by scripts/import_courses.py,
// then edited directly).
const courses = defineCollection({
  loader: glob({base: './src/content/courses', pattern: '*/course.json', generateId: ({entry}) => entry.split('/')[0]}),
  schema: z.object({
    key: z.string(), title: z.string(), brand: z.string(), description: z.string(), hero: z.string(), logo: z.string(),
    css: z.string(),                                   // stylesheet in public/css/: the course's own, or "house"
    accent: z.object({l: z.string(), d: z.string(), ink_d: z.string(), soft_l: z.string(), soft_d: z.string(), line_l: z.string(), line_d: z.string()}).optional(),
    parts: z.array(z.object({t: z.string(), m: z.array(z.string())})),
    modules: z.array(z.object({n: z.string(), name: z.string(), slug: z.string(), desc: z.string().optional()})),
    LV: z.array(z.number()), LVN: z.array(z.string()), GOAL: z.number().default(2),
    layer: z.number(), footnote: z.string().optional(), finalBadge: z.string().optional(),
    plans: z.record(z.string(), z.array(z.tuple([z.string(), z.string(), z.string()]))).optional(),
  }),
});

const lessons = defineCollection({
  loader: glob({base: './src/content/courses', pattern: '*/lessons/*.md', generateId: ({entry}) => entry.replace(/\.md$/, '')}),
  schema: z.object({
    id: z.string().regex(/^[0-9A-Z]{2}\.\d\d$/), course: z.string(), module: z.string(), slug: z.string(), title: z.string(),
    keyIdeas: z.array(z.string()).default([]), question: z.string().default(''), answer: z.string().default(''),
    rubric: z.array(z.string()).default([]),
    sources: z.array(z.object({title: z.string(), url: z.string().url(), note: z.string()})).default([]),
    practice: z.array(z.string()).default([]), tags: z.array(z.string()).default([]), tools: z.array(z.string()).default([]),
    topic: z.object({id: z.string(), title: z.string()}).optional(),       // ML Maths groups lessons into topics
    html: z.string().optional(), lead: z.string().optional(), mins: z.number().optional(),   // AI Research's HTML lessons
    blocks: z.array(z.any()).default([]),                                  // AI Research's notebook + quiz blocks
  }),
});

export const collections = { courses, lessons };
