import 'dotenv/config';
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import { parseMicrogravityCsv } from './src/lib/csvParser';
import {
  buildDeterministicSynthesis,
  detectLanguageFromText,
  evaluateHabitatRisk,
  parseNaturalLanguageQuestion,
} from './src/lib/fireSafetyEngine';
import type { HabitatQuery, Language, MicrogravityRecord, SISynthesis } from './src/types/fireSafety';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CSV_PATH = path.resolve(
  __dirname,
  'public/assets/aistudio/Tableau_Unique_Securite_Incendie_Microgravite_e4795b6a.csv'
);

let cachedRecords: MicrogravityRecord[] | null = null;
function loadRecords(): MicrogravityRecord[] {
  if (cachedRecords) return cachedRecords;
  const csvRaw = fs.readFileSync(CSV_PATH, 'utf8');
  cachedRecords = parseMicrogravityCsv(csvRaw);
  return cachedRecords;
}

function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function generateGeminiSynthesis(
  ai: GoogleGenAI,
  records: MicrogravityRecord[],
  query: HabitatQuery,
  lang: Language
): Promise<SISynthesis> {
  const assessment = evaluateHabitatRisk(records, query, lang);
  const topEvidence = assessment.rankedEvidence.slice(0, 8).map((item) => ({
    id: item.record.id,
    similarityScore: item.similarityScore,
    rowType: item.record.rowType,
    mission: item.record.mission,
    sourceId: item.record.sourceId,
    material: item.record.materialOriginal,
    flowDirection: item.record.flowDirection,
    o2Percent: item.record.o2Percent,
    pressureAtm: item.record.pressureAtm,
    velocityCmS: item.record.velocityCmS,
    vfMmS: item.record.vfMmS,
    kHotMm2S: item.record.kHotMm2S,
    coolFlame: item.record.coolFlame,
    loiO2Threshold: item.record.loiO2Threshold,
    uMinCmS: item.record.uMinCmS,
    observation: item.record.initialObservation,
    validityDomain: item.record.validityDomain,
    scientificAlert: item.record.scientificAlert,
  }));

  const targetLanguage = lang === 'fr' ? 'French' : 'English';
  const prompt = `You are a senior NASA Combustion & Spacecraft Fire Safety Super Intelligence (SI).
Answer the user's space habitat fire-risk question with strict scientific honesty, zero hallucinated numbers, and explicit epistemic humility.

USER QUESTION:
"${query.questionText || `Evaluate fire risk for ${query.materialKey === 'UNKNOWN_CUSTOM' ? query.customMaterialName : query.materialKey} at O2=${query.o2Percent}%, P=${query.pressureAtm} atm, U=${query.velocityCmS} cm/s (${query.flowDirection}), gravity=${query.gravity}.`}"

STRUCTURED HABITAT PARAMETERS:
- Material: ${query.materialKey} (${query.customMaterialName || 'standard'})
- O2: ${query.o2Percent}%, Pressure: ${query.pressureAtm} atm (pO2=${assessment.pO2Atm} atm), U: ${query.velocityCmS} cm/s (${query.flowDirection}), Gravity: ${query.gravity}

DETERMINISTIC AUDIT SUMMARY:
- Verdict: ${assessment.verdictLevel}, Known Material: ${assessment.isKnownMaterial}, Indicative LOI: ${assessment.loiO2Threshold ?? 'Unknown'}, Umin: ${assessment.uMinThreshold ?? 'Unknown'}
- Domain gaps: ${JSON.stringify(assessment.domainViolations)}
- Legacy divergence: ${JSON.stringify(assessment.legacyComparison.divergenceWarnings)}

TOP 8 RANKED NASA RECORDS (SOLE EVIDENCE):
${JSON.stringify(topEvidence)}

RULES:
1. Respond STRICTLY in ${targetLanguage} only. Even if the user question or NASA CSV fields contain French words (like "Tissu SIBAL", "Flamme froide", "Opposé"), translate all explanations, summaries, and observations into natural ${targetLanguage}.
2. NEVER invent measurements not present in the provided records.
3. Explicitly state what CANNOT be certified if material is unknown or parameters extrapolate beyond 0g / 1 atm.
4. Cite exact MG-xxx IDs in every finding.`;

  const response = await ai.models.generateContent({
    model: 'gemini-4-hydrogen',
    contents: prompt,
    config: {
      systemInstruction:
        'You are an empirically rigorous aerospace fire-safety analyst. Ground every numeric claim in the supplied MG-001..MG-039 records. Expose uncertainties and model limits honestly.',
      thinkingConfig: {
        thinkingLevel: ThinkingLevel.LOW,
      },
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          directAnswer: {
            type: Type.STRING,
            description: 'Direct, honest engineering answer to the user habitat question.',
          },
          honestRiskSummary: {
            type: Type.STRING,
            description: 'Concise synthesis of the governing combustion physics and domain limits.',
          },
          evidenceBreakdown: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: '3 to 4 concise evidence items citing MG-xxx IDs and exact measurements.',
          },
          whatWeDoNotKnow: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: '3 to 4 explicit scientific gaps, extrapolation limits, or legacy model flaws.',
          },
          safetyRecommendations: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: '3 concrete spacecraft / habitat fire-safety verification or mitigation actions.',
          },
          citedRecordIds: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: 'List of MG-xxx record IDs referenced in the synthesis.',
          },
        },
        required: [
          'directAnswer',
          'honestRiskSummary',
          'evidenceBreakdown',
          'whatWeDoNotKnow',
          'safetyRecommendations',
          'citedRecordIds',
        ],
      },
    },
  });

  const rawJson = (response.text ?? '').trim();
  const parsed = JSON.parse(rawJson) as Omit<SISynthesis, 'mode' | 'modelName' | 'generatedAt'>;

  return {
    ...parsed,
    mode: 'gemini',
    modelName: 'gemini-4-hydrogen',
    generatedAt: new Date().toISOString(),
  };
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '1mb' }));

  app.get('/api/health', (_req, res) => {
    const records = loadRecords();
    const hasGemini = Boolean(getGeminiClient());
    res.json({
      ok: true,
      recordCount: records.length,
      geminiConfigured: hasGemini,
    });
  });

  app.get('/api/dataset', (_req, res) => {
    try {
      const records = loadRecords();
      res.json({ records });
    } catch (error) {
      res.status(500).json({
        error: error instanceof Error ? error.message : 'Failed to load NASA dataset',
      });
    }
  });

  app.post('/api/analyze', async (req, res) => {
    try {
      const records = loadRecords();
      const body = req.body as {
        query?: Partial<HabitatQuery>;
        questionText?: string;
        language?: Language;
        useAi?: boolean;
      };

      const questionText = body.questionText?.trim() || body.query?.questionText?.trim() || '';
      const resolvedQuery: HabitatQuery = body.query?.materialKey
        ? {
            materialKey: body.query.materialKey,
            customMaterialName: body.query.customMaterialName ?? '',
            o2Percent: Number(body.query.o2Percent ?? 21),
            pressureAtm: Number(body.query.pressureAtm ?? 1),
            velocityCmS: Number(body.query.velocityCmS ?? 20),
            flowDirection: body.query.flowDirection ?? 'Concurrent',
            gravity: body.query.gravity ?? '0g',
            thicknessMm: body.query.thicknessMm ?? null,
            questionText,
          }
        : parseNaturalLanguageQuestion(questionText);

      const detectedLang: Language =
        body.language ?? (questionText ? detectLanguageFromText(questionText) : 'en');
      const assessment = evaluateHabitatRisk(records, resolvedQuery, detectedLang);

      const ai = body.useAi === false ? null : getGeminiClient();
      if (!ai) {
        const fallback = buildDeterministicSynthesis(
          assessment,
          resolvedQuery,
          detectedLang,
          'Deterministic NASA-PSI audit active (server Gemini API key not set or AI synthesis bypassed).'
        );
        return res.json({
          query: resolvedQuery,
          assessment,
          synthesis: fallback,
        });
      }

      try {
        const synthesis = await generateGeminiSynthesis(ai, records, resolvedQuery, detectedLang);
        return res.json({
          query: resolvedQuery,
          assessment,
          synthesis,
        });
      } catch (aiError) {
        const message = aiError instanceof Error ? aiError.message : 'Gemini request failed';
        const fallback = buildDeterministicSynthesis(
          assessment,
          resolvedQuery,
          detectedLang,
          `SI fallback to deterministic NASA-PSI audit (${message}).`
        );
        return res.json({
          query: resolvedQuery,
          assessment,
          synthesis: fallback,
        });
      }
    } catch (error) {
      res.status(400).json({
        error: error instanceof Error ? error.message : 'Invalid analysis request',
      });
    }
  });

  const isProd = process.env.NODE_ENV === 'production';
  if (isProd) {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('/{*splat}', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  const port = Number(process.env.PORT || 3000);
  app.listen(port, '0.0.0.0', () => {
    console.log(`NASA Microgravity Fire Safety SI listening on http://localhost:${port}`);
  });
}

startServer().catch((err) => {
  console.error('Fatal server startup error:', err);
  process.exit(1);
});
