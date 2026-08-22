#!/usr/bin/env node
"use strict";

const http = require("node:http");
const https = require("node:https");
const { StringDecoder } = require("node:string_decoder");

const DEFAULT_PORT = 48123;
const DEFAULT_MAX_BODY_BYTES = 64 * 1024 * 1024;

function parsePositiveInteger(value, fallback, name) {
  if (value === undefined || value === "") {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function normalizeUpstreamBaseUrl(value) {
  const url = new URL(value);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("NORTHGATE_UPSTREAM_BASE_URL must use http or https");
  }
  if (!url.pathname.endsWith("/")) {
    url.pathname += "/";
  }
  return url;
}

function customToolToFunction(tool) {
  const converted = {
    type: "function",
    name: String(tool.name || ""),
    description: typeof tool.description === "string" ? tool.description : "",
    parameters: {
      type: "object",
      properties: {
        input: {
          type: "string",
          description: `The ${String(tool.name || "custom tool")} raw input.`,
        },
      },
      required: ["input"],
      additionalProperties: false,
    },
  };
  for (const key of ["allowed_callers", "defer_loading", "cache_control"]) {
    if (tool[key] !== undefined) {
      converted[key] = tool[key];
    }
  }
  return converted;
}

function wrapCustomInput(input) {
  return JSON.stringify({ input: typeof input === "string" ? input : "" });
}

function unwrapFunctionArguments(argumentsText) {
  if (typeof argumentsText !== "string" || argumentsText.length === 0) {
    return "";
  }
  try {
    const parsed = JSON.parse(argumentsText);
    if (parsed && typeof parsed === "object") {
      if (typeof parsed.input === "string") {
        return parsed.input;
      }
      // Accept LiteLLM's stock wrapper while keeping `input` canonical upstream.
      if (typeof parsed.content === "string") {
        return parsed.content;
      }
    }
  } catch (_error) {
    // Preserve malformed arguments so Codex can surface the provider failure.
  }
  return argumentsText;
}

function coerceSchemaValue(value, schema) {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    return value;
  }
  const declaredTypes = Array.isArray(schema.type) ? schema.type : [schema.type];
  if (
    typeof value === "string" &&
    declaredTypes.some((type) => type === "integer" || type === "number") &&
    /^-?(?:\d+\.?\d*|\.\d+)$/.test(value.trim())
  ) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  if (typeof value === "string" && declaredTypes.includes("boolean")) {
    if (value === "true") {
      return true;
    }
    if (value === "false") {
      return false;
    }
  }
  if (Array.isArray(value) && declaredTypes.includes("array")) {
    return value.map((item) => coerceSchemaValue(item, schema.items));
  }
  if (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    declaredTypes.includes("object")
  ) {
    const properties =
      schema.properties && typeof schema.properties === "object"
        ? schema.properties
        : {};
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        coerceSchemaValue(item, properties[key]),
      ]),
    );
  }
  return value;
}

function normalizeFunctionArguments(argumentsText, tool) {
  if (typeof argumentsText !== "string" || !tool) {
    return argumentsText;
  }
  try {
    const parsed = JSON.parse(argumentsText);
    return JSON.stringify(coerceSchemaValue(parsed, tool.parameters));
  } catch (_error) {
    return argumentsText;
  }
}

function isCodeModeSource(input) {
  return (
    /\btools\.[A-Za-z_][A-Za-z0-9_]*\s*\(/.test(input) ||
    /\b(?:text|image|generatedImage|store|load|notify|yield_control)\s*\(/.test(input) ||
    /^\s*\/\/\s*@exec:/.test(input)
  );
}

function normalizeCustomInput(name, input) {
  if (name !== "exec" || typeof input !== "string") {
    return input;
  }
  const trimmed = input.trim();
  if (!trimmed || isCodeModeSource(trimmed)) {
    return input;
  }

  let commandArgs;
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object" && typeof parsed.cmd === "string") {
      commandArgs = parsed;
    }
  } catch (_error) {
    // A non-JSON raw input from N2 is treated as a shell command below.
  }
  if (!commandArgs) {
    commandArgs = { cmd: input };
  }
  for (const key of ["yield_time_ms", "max_output_tokens"]) {
    if (typeof commandArgs[key] === "string" && /^\d+$/.test(commandArgs[key])) {
      commandArgs[key] = Number(commandArgs[key]);
    }
  }
  return [
    `const result = await tools.exec_command(${JSON.stringify(commandArgs)});`,
    "text(result.output ?? JSON.stringify(result));",
  ].join("\n");
}

function transformToolChoice(toolChoice, customNames) {
  if (!toolChoice || typeof toolChoice !== "object" || Array.isArray(toolChoice)) {
    return toolChoice;
  }
  const converted = { ...toolChoice };
  if (converted.type === "custom" && customNames.has(converted.name)) {
    converted.type = "function";
  }
  if (Array.isArray(converted.tools)) {
    converted.tools = converted.tools.map((tool) =>
      tool && typeof tool === "object"
        ? transformToolChoice(tool, customNames)
        : tool,
    );
  }
  return converted;
}

function transformInputItem(item, customNames) {
  if (!item || typeof item !== "object" || Array.isArray(item)) {
    return item;
  }
  if (item.type === "custom_tool_call" && customNames.has(item.name)) {
    const converted = {
      ...item,
      type: "function_call",
      arguments: wrapCustomInput(item.input),
    };
    delete converted.input;
    delete converted.namespace;
    return converted;
  }
  if (item.type === "custom_tool_call_output") {
    const converted = { ...item, type: "function_call_output" };
    delete converted.name;
    delete converted.namespace;
    return converted;
  }
  return item;
}

function transformRequestBody(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw new Error("Responses request body must be a JSON object");
  }

  const tools = Array.isArray(body.tools) ? body.tools : [];
  const customTools = new Map();
  const functionNames = new Set();
  const functionTools = new Map();
  for (const tool of tools) {
    if (!tool || typeof tool !== "object") {
      continue;
    }
    if (tool.type === "custom") {
      const name = String(tool.name || "");
      if (!name) {
        throw new Error("Custom tools must have a non-empty name");
      }
      customTools.set(name, { ...tool });
    } else if (tool.type === "function" && typeof tool.name === "string") {
      functionNames.add(tool.name);
      functionTools.set(tool.name, { ...tool });
    } else if (tool.type === "namespace") {
      throw new Error(
        "Namespace tools are not supported by the NorthGate bridge; disable multi-agent tools",
      );
    }
  }
  for (const name of customTools.keys()) {
    if (functionNames.has(name)) {
      throw new Error(`Tool name ${name} is declared as both custom and function`);
    }
  }

  const customNames = new Set(customTools.keys());
  const converted = { ...body };
  if (Array.isArray(body.tools)) {
    converted.tools = body.tools.map((tool) =>
      tool && typeof tool === "object" && tool.type === "custom"
        ? customToolToFunction(tool)
        : tool,
    );
  }
  if (Array.isArray(body.input)) {
    converted.input = body.input.map((item) => transformInputItem(item, customNames));
  }
  converted.tool_choice = transformToolChoice(body.tool_choice, customNames);

  return { body: converted, customNames, customTools, functionTools };
}

function transformCallItem(item, context) {
  if (!item || typeof item !== "object" || Array.isArray(item)) {
    return item;
  }
  if (item.type !== "function_call") {
    return item;
  }
  const normalizedItem = {
    ...item,
    arguments: normalizeFunctionArguments(
      item.arguments,
      context.functionTools.get(item.name),
    ),
  };
  if (!context.customNames.has(item.name)) {
    return normalizedItem;
  }
  const converted = {
    ...normalizedItem,
    type: "custom_tool_call",
    input: normalizeCustomInput(
      item.name,
      unwrapFunctionArguments(normalizedItem.arguments),
    ),
  };
  delete converted.arguments;
  delete converted.namespace;
  return converted;
}

function transformResponseObject(response, context) {
  if (!response || typeof response !== "object" || Array.isArray(response)) {
    return response;
  }
  const converted = { ...response };
  if (Array.isArray(response.output)) {
    converted.output = response.output.map((item) => transformCallItem(item, context));
  }
  if (Array.isArray(response.tools)) {
    converted.tools = response.tools.map((tool) => {
      if (
        tool &&
        typeof tool === "object" &&
        tool.type === "function" &&
        context.customTools.has(tool.name)
      ) {
        return { ...context.customTools.get(tool.name) };
      }
      return tool;
    });
  }
  return converted;
}

function callKey(itemId, callId) {
  return String(itemId || callId || "");
}

function registerCustomCall(item, context) {
  if (!item || typeof item !== "object" || !context.customNames.has(item.name)) {
    return false;
  }
  const record = {
    name: item.name,
    arguments: typeof item.arguments === "string" ? item.arguments : "",
  };
  for (const key of [callKey(item.id), callKey(item.call_id)]) {
    if (key) {
      context.calls.set(key, record);
    }
  }
  return true;
}

function transformStreamEvent(event, context) {
  if (!event || typeof event !== "object" || Array.isArray(event)) {
    return [event];
  }

  if (event.response && typeof event.response === "object") {
    event = { ...event, response: transformResponseObject(event.response, context) };
  }

  if (
    event.type === "response.output_item.added" ||
    event.type === "response.output_item.done"
  ) {
    const originalItem = event.item;
    registerCustomCall(originalItem, context);
    return [{ ...event, item: transformCallItem(originalItem, context) }];
  }

  if (
    event.type === "response.function_call_arguments.delta" ||
    event.type === "response.function_call_arguments.done"
  ) {
    const key = callKey(event.item_id, event.call_id);
    const record = context.calls.get(key);
    if (!record) {
      if (event.type === "response.function_call_arguments.done") {
        return [
          {
            ...event,
            arguments: normalizeFunctionArguments(
              event.arguments,
              context.functionTools.get(event.name),
            ),
          },
        ];
      }
      return [event];
    }

    if (event.type === "response.function_call_arguments.delta") {
      if (typeof event.delta === "string") {
        record.arguments += event.delta;
      }
      return [];
    }

    const argumentsText =
      typeof event.arguments === "string" ? event.arguments : record.arguments;
    const input = normalizeCustomInput(
      record.name,
      unwrapFunctionArguments(argumentsText),
    );
    const common = {
      item_id: event.item_id,
      output_index: event.output_index,
    };
    const converted = [];
    if (input) {
      converted.push({
        ...common,
        type: "response.custom_tool_call_input.delta",
        delta: input,
      });
    }
    converted.push({
      ...common,
      type: "response.custom_tool_call_input.done",
      input,
    });
    return converted;
  }

  return [event];
}

function assignSequenceNumber(event, context) {
  if (event && typeof event === "object" && String(event.type || "").startsWith("response.")) {
    return { ...event, sequence_number: context.nextSequence++ };
  }
  return event;
}

function serializeSseEvent(event) {
  return `event: ${String(event.type || "message")}\ndata: ${JSON.stringify(event)}\n\n`;
}

function transformSseFrame(frame, context) {
  const lines = frame.split(/\r?\n/);
  const dataLines = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""));
  if (dataLines.length === 0) {
    return frame ? `${frame}\n\n` : "";
  }
  const data = dataLines.join("\n");
  if (data === "[DONE]") {
    return "data: [DONE]\n\n";
  }
  let event;
  try {
    event = JSON.parse(data);
  } catch (_error) {
    return `${frame}\n\n`;
  }
  return transformStreamEvent(event, context)
    .map((item) => serializeSseEvent(assignSequenceNumber(item, context)))
    .join("");
}

function copyUpstreamHeaders(headers, bodyLength) {
  const result = {};
  for (const [name, value] of Object.entries(headers)) {
    const lower = name.toLowerCase();
    if (
      value !== undefined &&
      !["host", "content-length", "connection", "accept-encoding"].includes(lower)
    ) {
      result[lower] = value;
    }
  }
  result["content-type"] = "application/json";
  result["content-length"] = String(bodyLength);
  result["accept-encoding"] = "identity";
  return result;
}

function copyDownstreamHeaders(headers, isStreaming) {
  const result = {};
  for (const [name, value] of Object.entries(headers)) {
    const lower = name.toLowerCase();
    if (
      value !== undefined &&
      !["content-length", "content-encoding", "transfer-encoding", "connection"].includes(lower)
    ) {
      result[lower] = value;
    }
  }
  result["content-type"] = isStreaming
    ? "text/event-stream; charset=utf-8"
    : "application/json";
  result["cache-control"] = "no-cache";
  return result;
}

function readRequestBody(request, maxBodyBytes) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > maxBodyBytes) {
        reject(Object.assign(new Error("Request body is too large"), { statusCode: 413 }));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => resolve(Buffer.concat(chunks)));
    request.on("error", reject);
  });
}

function requestUpstream(url, headers, body) {
  const client = url.protocol === "https:" ? https : http;
  return new Promise((resolve, reject) => {
    const request = client.request(
      url,
      { method: "POST", headers: copyUpstreamHeaders(headers, body.length) },
      resolve,
    );
    request.on("error", reject);
    request.end(body);
  });
}

function collectResponse(response, maxBodyBytes) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    response.on("data", (chunk) => {
      size += chunk.length;
      if (size > maxBodyBytes) {
        reject(new Error("Upstream response body is too large"));
        response.destroy();
        return;
      }
      chunks.push(chunk);
    });
    response.on("end", () => resolve(Buffer.concat(chunks)));
    response.on("error", reject);
  });
}

async function proxyResponses(request, response, options) {
  const rawBody = await readRequestBody(request, options.maxBodyBytes);
  let parsedBody;
  try {
    parsedBody = JSON.parse(rawBody.toString("utf8"));
  } catch (_error) {
    const error = new Error("Request body must be valid JSON");
    error.statusCode = 400;
    throw error;
  }

  let transformed;
  try {
    transformed = transformRequestBody(parsedBody);
  } catch (error) {
    error.statusCode = 400;
    throw error;
  }

  const upstreamBody = Buffer.from(JSON.stringify(transformed.body));
  const upstreamUrl = new URL("responses", options.upstreamBaseUrl);
  const upstreamResponse = await requestUpstream(
    upstreamUrl,
    request.headers,
    upstreamBody,
  );
  const contentType = String(upstreamResponse.headers["content-type"] || "");
  const isStreaming = contentType.includes("text/event-stream");
  response.writeHead(
    upstreamResponse.statusCode || 502,
    copyDownstreamHeaders(upstreamResponse.headers, isStreaming),
  );

  const context = {
    customNames: transformed.customNames,
    customTools: transformed.customTools,
    functionTools: transformed.functionTools,
    calls: new Map(),
    nextSequence: 0,
  };

  if (!isStreaming) {
    const body = await collectResponse(upstreamResponse, options.maxBodyBytes);
    try {
      const parsed = JSON.parse(body.toString("utf8"));
      response.end(JSON.stringify(transformResponseObject(parsed, context)));
    } catch (_error) {
      response.end(body);
    }
    return;
  }

  const decoder = new StringDecoder("utf8");
  let buffer = "";
  for await (const chunk of upstreamResponse) {
    buffer += decoder.write(chunk);
    while (true) {
      const match = /\r?\n\r?\n/.exec(buffer);
      if (!match) {
        break;
      }
      const frame = buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
      const output = transformSseFrame(frame, context);
      if (output) {
        response.write(output);
      }
    }
  }
  buffer += decoder.end();
  if (buffer.trim()) {
    response.write(transformSseFrame(buffer.trimEnd(), context));
  }
  response.end();
}

function sendJson(response, statusCode, payload) {
  const body = Buffer.from(JSON.stringify(payload));
  response.writeHead(statusCode, {
    "content-type": "application/json",
    "content-length": String(body.length),
  });
  response.end(body);
}

function createBridgeServer(options) {
  return http.createServer(async (request, response) => {
    try {
      if (request.method === "GET" && request.url === "/health") {
        sendJson(response, 200, { status: "ok" });
        return;
      }
      if (
        request.method !== "POST" ||
        !["/responses", "/v1/responses"].includes(String(request.url || "").split("?", 1)[0])
      ) {
        sendJson(response, 404, { error: { message: "Not found", type: "invalid_request_error" } });
        return;
      }
      await proxyResponses(request, response, options);
    } catch (error) {
      if (response.headersSent) {
        response.destroy(error);
        return;
      }
      const statusCode = Number(error.statusCode) || 502;
      sendJson(response, statusCode, {
        error: {
          message: error.message || "NorthGate bridge failure",
          type: statusCode < 500 ? "invalid_request_error" : "server_error",
        },
      });
    }
  });
}

function main() {
  const upstreamValue = process.env.NORTHGATE_UPSTREAM_BASE_URL;
  if (!upstreamValue) {
    throw new Error("NORTHGATE_UPSTREAM_BASE_URL is required");
  }
  const options = {
    upstreamBaseUrl: normalizeUpstreamBaseUrl(upstreamValue),
    port: parsePositiveInteger(
      process.env.NORTHGATE_BRIDGE_PORT,
      DEFAULT_PORT,
      "NORTHGATE_BRIDGE_PORT",
    ),
    maxBodyBytes: parsePositiveInteger(
      process.env.NORTHGATE_BRIDGE_MAX_BODY_BYTES,
      DEFAULT_MAX_BODY_BYTES,
      "NORTHGATE_BRIDGE_MAX_BODY_BYTES",
    ),
  };
  const server = createBridgeServer(options);
  server.listen(options.port, "127.0.0.1", () => {
    process.stdout.write(`NorthGate Responses bridge listening on 127.0.0.1:${options.port}\n`);
  });
  const shutdown = () => server.close(() => process.exit(0));
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

module.exports = {
  createBridgeServer,
  customToolToFunction,
  transformRequestBody,
  transformResponseObject,
  transformStreamEvent,
  normalizeCustomInput,
  normalizeFunctionArguments,
  unwrapFunctionArguments,
};

if (require.main === module) {
  main();
}
