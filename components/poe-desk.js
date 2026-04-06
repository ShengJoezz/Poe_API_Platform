"use client";

import { startTransition, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  FolderArchive,
  FolderPlus,
  MessageSquare,
  Paperclip,
  PanelLeft,
  Plus,
  RefreshCw,
  Settings,
  Square,
  SquarePen,
  Trash2,
  Wallet,
  X,
} from "lucide-react";

const STORAGE_KEY = "poe-gpt-5-4-pro-chat-settings";
const AUTO_SYNC_INTERVAL_MS = 5 * 60 * 1000;

const MODEL_PRESETS = {
  gpt54pro: {
    model: "gpt-5.4-pro",
    display: "GPT-5.4-Pro",
    maxOutputTokens: 128000,
    defaultExtraBody: {
      web_search: true,
      reasoning_effort: "xhigh",
      verbosity: "high",
    },
  },
  claudeopus46: {
    model: "claude-opus-4.6",
    display: "Claude-Opus-4.6",
    maxOutputTokens: 128000,
    defaultExtraBody: {
      web_search: true,
      output_effort: "max",
    },
  },
  geminiflashlite: {
    model: "gemini-2.0-flash-lite",
    display: "Gemini-2.0-Flash-Lite",
    maxOutputTokens: 8192,
    defaultExtraBody: null,
  },
};

function generateConversationId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}`;
}

function inferPresetFromModel(modelName) {
  const normalized = String(modelName || "").trim().toLowerCase();
  for (const [presetKey, preset] of Object.entries(MODEL_PRESETS)) {
    if (
      preset.model &&
      (preset.model.toLowerCase() === normalized || String(preset.display || "").toLowerCase() === normalized)
    ) {
      return presetKey;
    }
  }
  return "gpt54pro";
}

function parseOptionalNumber(rawValue, fieldLabel) {
  if (!rawValue) {
    return undefined;
  }
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${fieldLabel}需为数字`);
  }
  return parsed;
}

function parseStoredExtraBody(rawValue) {
  if (!rawValue) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(rawValue);
    return parsed && !Array.isArray(parsed) && typeof parsed === "object" ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function decodeMimeTypeFromDataUrl(dataUrl) {
  const match = /^data:([^;,]+)[;,]/.exec(String(dataUrl || ""));
  return match ? match[1] : "application/octet-stream";
}

function attachmentExtension(name) {
  const pieces = String(name || "").split(".");
  return pieces.length > 1 ? pieces.at(-1).slice(0, 4).toUpperCase() : "FILE";
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[index]}`;
}

function getTotalAttachmentBytes(attachments) {
  return attachments.reduce((total, attachment) => total + (attachment.size || 0), 0);
}

function attachmentPartToDisplayAttachment(part, index) {
  if (part.type === "image_url" && part.image_url?.url) {
    const dataUrl = part.image_url.url;
    return {
      id: `content-image-${index}`,
      name: `image-${index + 1}`,
      kind: "image",
      mimeType: decodeMimeTypeFromDataUrl(dataUrl),
      size: 0,
      dataUrl,
    };
  }

  if (part.type === "file" && part.file?.file_data) {
    const dataUrl = part.file.file_data;
    const mimeType = decodeMimeTypeFromDataUrl(dataUrl);
    return {
      id: `content-file-${index}`,
      name: part.file.filename || `attachment-${index + 1}`,
      kind: mimeType.startsWith("image/") ? "image" : "file",
      mimeType,
      size: 0,
      dataUrl,
    };
  }

  return null;
}

function extractStructuredContent(content) {
  if (typeof content === "string") {
    return { text: content, attachments: [] };
  }

  const textParts = [];
  const attachments = [];
  if (Array.isArray(content)) {
    content.forEach((part, index) => {
      if (part?.type === "text" && part.text) {
        textParts.push(String(part.text));
      }
      const attachment = attachmentPartToDisplayAttachment(part, index);
      if (attachment) {
        attachments.push(attachment);
      }
    });
  }

  return {
    text: textParts.join("\n\n"),
    attachments,
  };
}

function humanizeStatus(status) {
  const mapping = {
    queued: "排队中",
    streaming: "生成中",
    partial_saved: "已保存",
    completed: "已完成",
    timed_out: "超时",
    failed: "失败",
  };
  return mapping[status] || status || "local";
}

function humanizeReconcileStatus(status) {
  const mapping = {
    pending: "待核对",
    matched: "已对账",
    unmatched: "有差异",
  };
  return mapping[status] || status || "待核对";
}

function cloneAttachment(attachment) {
  return { ...attachment };
}

function attachmentToContentPart(attachment) {
  if (attachment.kind === "image") {
    return {
      type: "image_url",
      image_url: { url: attachment.dataUrl },
    };
  }

  return {
    type: "file",
    file: {
      filename: attachment.name,
      file_data: attachment.dataUrl,
    },
  };
}

function buildUserContent(rawUserText, pendingAttachments) {
  const userText = String(rawUserText || "").trim();
  const parts = [];

  if (userText) {
    parts.push({ type: "text", text: userText });
  }

  pendingAttachments.forEach((attachment) => {
    parts.push(attachmentToContentPart(attachment));
  });

  if (parts.length === 0) {
    throw new Error("请输入消息或添加附件");
  }

  if (parts.length === 1 && parts[0].type === "text") {
    return parts[0].text;
  }

  return parts;
}

async function fileToAttachment(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      resolve({
        id: globalThis.crypto?.randomUUID?.() || `${Date.now()}`,
        name: file.name,
        size: file.size,
        mimeType: file.type || "application/octet-stream",
        kind: file.type.startsWith("image/") ? "image" : "file",
        dataUrl: String(reader.result),
      });
    reader.onerror = () => reject(new Error("读取失败"));
    reader.readAsDataURL(file);
  });
}

function renderAttachmentPreview(attachment) {
  return (
    <div className="bubble-attachment" key={attachment.id}>
      {attachment.kind === "image" ? (
        <img src={attachment.dataUrl} alt={attachment.name} />
      ) : (
        <span className="bubble-attachment-icon">{attachmentExtension(attachment.name)}</span>
      )}
      <span className="bubble-attachment-name">{attachment.name}</span>
    </div>
  );
}

export default function PoeDesk() {
  const [hydrated, setHydrated] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [rememberKey, setRememberKey] = useState(false);
  const [modelPreset, setModelPreset] = useState("gpt54pro");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [temperature, setTemperature] = useState("");
  const [maxTokens, setMaxTokens] = useState("");
  const [minBalanceGuard, setMinBalanceGuard] = useState("");
  const [maxAttachmentMb, setMaxAttachmentMb] = useState("15");
  const [webSearch, setWebSearch] = useState(true);
  const [reasoningEffort, setReasoningEffort] = useState("xhigh");
  const [verbosity, setVerbosity] = useState("high");
  const [outputEffort, setOutputEffort] = useState("max");
  const [conversationId, setConversationId] = useState("");

  const [messages, setMessages] = useState([]);
  const [conversationMeta, setConversationMeta] = useState(null);
  const [folders, setFolders] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [recentRequests, setRecentRequests] = useState([]);
  const [auditSummary, setAuditSummary] = useState({});
  const [pendingAttachments, setPendingAttachments] = useState([]);
  const [currentRequestId, setCurrentRequestId] = useState(null);
  const [statusText, setStatusText] = useState("已就绪");
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [activeFolderId, setActiveFolderId] = useState("__all__");
  const [draggedConversationId, setDraggedConversationId] = useState(null);
  const [dragOverFolderId, setDragOverFolderId] = useState(null);
  const [folderBusy, setFolderBusy] = useState(false);
  const [userInput, setUserInput] = useState("");

  const readerRef = useRef(null);
  const userInputRef = useRef(null);
  const fileInputRef = useRef(null);
  const chatLogRef = useRef(null);

  const currentModelConfig = MODEL_PRESETS[modelPreset] || MODEL_PRESETS.gpt54pro;
  const currentModelId = currentModelConfig.model;
  const currentModelLabel = currentModelConfig.display;
  const currentMaxOutputTokens = currentModelConfig.maxOutputTokens;
  const latestBalance = auditSummary?.latestBalance;
  const activeFolder = folders.find((folder) => folder.id === activeFolderId) || null;
  const currentConversationCost = recentRequests
    .filter((request) => request.conversationId === conversationId)
    .reduce((sum, request) => sum + (Number(request.usageCostPoints) || 0), 0);
  const unfiledConversations = conversations.filter((item) => !item.folderId);
  const visibleConversations =
    activeFolderId === "__all__"
      ? conversations
      : activeFolderId === "__unfiled__"
        ? unfiledConversations
        : conversations.filter((item) => item.folderId === activeFolderId);

  useEffect(() => {
    if (chatLogRef.current) {
      chatLogRef.current.scrollTop = chatLogRef.current.scrollHeight;
    }
  }, [messages, pendingAttachments]);

  async function fetchState() {
    if (!conversationId) {
      return;
    }

    const response = await fetch(`/api/state?conversation_id=${encodeURIComponent(conversationId)}`, {
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    startTransition(() => {
      setConversationMeta(payload.conversation || null);
      setFolders(payload.folders || []);
      setConversations(payload.conversations || []);
      setMessages(payload.transcript || []);
      setRecentRequests(payload.recentRequests || []);
      setAuditSummary(payload.summary || {});
    });
  }

  async function syncAudit() {
    if (!apiKey.trim()) {
      throw new Error("请先填写 API Key");
    }

    const response = await fetch("/api/sync", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ apiKey: apiKey.trim() }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || "同步失败");
    }

    return response.json();
  }

  async function refreshAll({ syncRemote = false } = {}) {
    if (syncRemote && apiKey.trim()) {
      await syncAudit();
    }
    await fetchState();
  }

  async function readSseStream(reader, optimisticAssistantId) {
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        return;
      }

      buffer += decoder.decode(value, { stream: true });
      while (buffer.includes("\n\n")) {
        const boundaryIndex = buffer.indexOf("\n\n");
        const eventBlock = buffer.slice(0, boundaryIndex);
        buffer = buffer.slice(boundaryIndex + 2);

        const lines = eventBlock.split("\n");
        const eventLine = lines.find((line) => line.startsWith("event:"));
        const dataLine = lines.find((line) => line.startsWith("data:"));
        const eventName = eventLine ? eventLine.slice(6).trim() : "message";
        const data = dataLine ? JSON.parse(dataLine.slice(5).trim()) : {};

        if (eventName === "ready") {
          setCurrentRequestId(data.requestId);
          setBusy(true);
          setStatusText("生成中...");
        }

        if (eventName === "delta" && typeof data.content === "string") {
          setMessages((current) =>
            current.map((message) =>
              message.messageId === optimisticAssistantId
                ? { ...message, content: `${message.content || ""}${data.content}`, status: "streaming" }
                : message
            )
          );
        }

        if (eventName === "reasoning" && typeof data.content === "string") {
          setMessages((current) =>
            current.map((message) =>
              message.messageId === optimisticAssistantId
                ? {
                    ...message,
                    reasoningContent: `${message.reasoningContent || ""}${data.content}`,
                    status: "streaming",
                  }
                : message
            )
          );
        }

        if (eventName === "usage") {
          setStatusText("对账中...");
        }

        if (eventName === "cancelled") {
          setStatusText("已停止");
        }

        if (eventName === "error") {
          throw new Error(data.message || "流式响应失败");
        }

        if (eventName === "completed") {
          return;
        }
      }
    }
  }

  function appendSystemMessage(text) {
    setMessages((current) => [
      ...current,
      {
        messageId: `${generateConversationId()}:system`,
        role: "system",
        content: text,
        status: "local",
      },
    ]);
  }

  async function handleAddAttachments(fileList) {
    const list = Array.from(fileList || []);
    if (!list.length) {
      return;
    }

    const maxMb = parseOptionalNumber(maxAttachmentMb.trim(), "上限") || 15;
    const nextTotalBytes = getTotalAttachmentBytes(pendingAttachments) + list.reduce((sum, file) => sum + file.size, 0);
    if (nextTotalBytes > maxMb * 1024 * 1024) {
      throw new Error(`超出 ${maxMb}MB`);
    }

    setStatusText("读取附件...");
    const attachments = await Promise.all(list.map(fileToAttachment));
    setPendingAttachments((current) => [...current, ...attachments]);
    setStatusText("已就绪");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (readerRef.current) {
      return;
    }

    const savedAttachments = pendingAttachments.map(cloneAttachment);
    const rawUserText = userInput;

    try {
      const userContent = buildUserContent(rawUserText, pendingAttachments);
      const optimisticAssistantId = `${generateConversationId()}:assistant`;
      const requestedMaxTokens = parseOptionalNumber(maxTokens, "输出上限");
      const effectiveExtraBody =
        currentModelId === "gpt-5.4-pro"
          ? {
              web_search: webSearch,
              reasoning_effort: reasoningEffort,
              verbosity,
            }
          : currentModelId === "claude-opus-4.6"
            ? {
                web_search: webSearch,
                output_effort: outputEffort,
              }
            : undefined;

      setBusy(true);
      setStatusText("发送中...");
      setMessages((current) => [
        ...current,
        {
          messageId: `${generateConversationId()}:user`,
          role: "user",
          content: userContent,
          status: "queued",
        },
        {
          messageId: optimisticAssistantId,
          role: "assistant",
          content: "",
          reasoningContent: "",
          status: "streaming",
        },
      ]);

      setUserInput("");
      setPendingAttachments([]);
      if (userInputRef.current) {
        userInputRef.current.style.height = "auto";
      }

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          apiKey: apiKey.trim(),
          conversationId,
          model: currentModelId,
          systemPrompt,
          userContent,
          minBalanceGuard: parseOptionalNumber(minBalanceGuard, "最低余额"),
          temperature: parseOptionalNumber(temperature, "温度"),
          maxOutputTokens:
            requestedMaxTokens !== undefined
              ? Math.min(requestedMaxTokens, currentMaxOutputTokens)
              : currentMaxOutputTokens,
          extraBody: effectiveExtraBody,
        }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `HTTP ${response.status}`);
      }

      if (!response.body) {
        throw new Error("响应为空");
      }

      const reader = response.body.getReader();
      readerRef.current = reader;
      await readSseStream(reader, optimisticAssistantId);
      await refreshAll({ syncRemote: false });
      setStatusText("已就绪");
    } catch (error) {
      setStatusText("发送失败");
      setUserInput(rawUserText);
      setPendingAttachments(savedAttachments);
      await fetchState().catch(() => {});
      appendSystemMessage(`错误：${error.message}`);
    } finally {
      readerRef.current = null;
      setCurrentRequestId(null);
      setBusy(false);
      userInputRef.current?.focus();
    }
  }

  async function stopCurrentRequest() {
    if (!currentRequestId) {
      return;
    }

    setStatusText("停止中...");
    await fetch(`/api/requests/${currentRequestId}/cancel`, { method: "POST" });
  }

  function loadConversation(targetConversation) {
    if (!targetConversation?.id) {
      return;
    }

    setConversationId(targetConversation.id);
    setMessages([]);
    setPendingAttachments([]);

    const nextModel = targetConversation.model || currentModelId;
    const inferredPreset = inferPresetFromModel(nextModel);
    setModelPreset(inferredPreset);
    setSystemPrompt(typeof targetConversation.systemPrompt === "string" ? targetConversation.systemPrompt : "");
    setStatusText("已载入");
    setSidebarOpen(false);
    setSettingsOpen(false);
  }

  function startNewConversation() {
    const nextConversationId = generateConversationId();
    setConversationId(nextConversationId);
    setConversationMeta({
      id: nextConversationId,
      title: "新对话",
      folderId: null,
      systemPrompt,
      model: currentModelId,
    });
    setMessages([]);
    setPendingAttachments([]);
    setCurrentRequestId(null);
    setUserInput("");
    setStatusText("已就绪");
    setSidebarOpen(false);
    setSettingsOpen(false);
  }

  function exportEvidence(requestId) {
    const link = document.createElement("a");
    link.href = `/api/requests/${requestId}/evidence.zip`;
    link.download = `poe-evidence-${requestId}.zip`;
    document.body.append(link);
    link.click();
    link.remove();
  }

  async function deleteRequest(requestId) {
    if (!globalThis.confirm("确定要删除这条记录吗？")) {
      return;
    }

    try {
      setStatusText("删除中...");
      const response = await fetch(`/api/requests/${requestId}`, { method: "DELETE" });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || "删除失败");
      }
      await fetchState();
      setStatusText("已删除");
    } catch (error) {
      setStatusText("删除报错");
      appendSystemMessage(`删除错误：${error.message}`);
    }
  }

  async function createFolder(rawName) {
    const folderName = String(rawName || "").trim();
    if (!folderName) {
      return;
    }

    try {
      setFolderBusy(true);
      setStatusText("创建文件夹...");
      const response = await fetch("/api/folders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: folderName }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || "创建失败");
      }
      await fetchState();
      setStatusText("文件夹已创建");
    } catch (error) {
      setStatusText("创建失败");
      appendSystemMessage(`文件夹错误：${error.message}`);
    } finally {
      setFolderBusy(false);
    }
  }

  async function handleCreateFolderPrompt() {
    const folderName = globalThis.prompt("请输入文件夹名称");
    if (folderName === null) {
      return;
    }
    await createFolder(folderName);
  }

  async function moveConversationToFolder(targetConversationId, targetFolderId) {
    if (!targetConversationId) {
      return;
    }

    try {
      setFolderBusy(true);
      setStatusText(targetFolderId ? "归档中..." : "移出文件夹...");
      const targetConversation = conversations.find((item) => item.id === targetConversationId);
      const normalizedFolderId = targetFolderId === "__unfiled__" ? null : targetFolderId || null;

      if ((targetConversation?.folderId || null) === normalizedFolderId) {
        setStatusText("已就绪");
        return;
      }

      const response = await fetch(`/api/conversations/${encodeURIComponent(targetConversationId)}/folder`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          folderId: normalizedFolderId,
          model: targetConversation?.model || currentModelId,
          systemPrompt: targetConversation?.systemPrompt || systemPrompt,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || "归档失败");
      }
      await fetchState();
      setStatusText(normalizedFolderId ? "已归档" : "已移出文件夹");
    } catch (error) {
      setStatusText("归档失败");
      appendSystemMessage(`归档错误：${error.message}`);
    } finally {
      setFolderBusy(false);
      setDragOverFolderId(null);
      setDraggedConversationId(null);
    }
  }

  async function deleteFolder(folder) {
    if (!folder?.id) {
      return;
    }

    if (!globalThis.confirm(`删除“${folder.name}”后，其中的会话会回到未归档。确定继续吗？`)) {
      return;
    }

    try {
      setFolderBusy(true);
      setStatusText("删除文件夹...");
      const response = await fetch(`/api/folders/${encodeURIComponent(folder.id)}`, {
        method: "DELETE",
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || "删除失败");
      }
      await fetchState();
      setStatusText("文件夹已删除");
    } catch (error) {
      setStatusText("删除失败");
      appendSystemMessage(`文件夹错误：${error.message}`);
    } finally {
      setFolderBusy(false);
    }
  }

  async function handleManualSync() {
    try {
      setStatusText("同步中...");
      await refreshAll({ syncRemote: true });
      setStatusText("已同步");
    } catch (error) {
      setStatusText("同步失败");
      appendSystemMessage(`错误：${error.message}`);
    }
  }

  useEffect(() => {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      try {
        const settings = JSON.parse(raw);
        const legacyExtraBody = parseStoredExtraBody(settings.extraBody);
        setApiKey(settings.apiKey || "");
        setRememberKey(Boolean(settings.rememberKey));
        const inferredPreset = MODEL_PRESETS[settings.modelPreset]
          ? settings.modelPreset
          : inferPresetFromModel(settings.model);
        setModelPreset(inferredPreset);
        setSystemPrompt(settings.systemPrompt || "");
        setTemperature(settings.temperature || "");
        setMaxTokens(settings.maxTokens || "");
        setMinBalanceGuard(settings.minBalanceGuard || "");
        setMaxAttachmentMb(settings.maxAttachmentMb || "15");
        setWebSearch(
          typeof settings.webSearch === "boolean"
            ? settings.webSearch
            : typeof legacyExtraBody?.web_search === "boolean"
              ? legacyExtraBody.web_search
              : true
        );
        setReasoningEffort(
          typeof settings.reasoningEffort === "string"
            ? settings.reasoningEffort
            : typeof legacyExtraBody?.reasoning_effort === "string"
              ? legacyExtraBody.reasoning_effort
              : "xhigh"
        );
        setVerbosity(
          typeof settings.verbosity === "string"
            ? settings.verbosity
            : typeof legacyExtraBody?.verbosity === "string"
              ? legacyExtraBody.verbosity
              : "high"
        );
        setOutputEffort(
          typeof settings.outputEffort === "string"
            ? settings.outputEffort
            : typeof legacyExtraBody?.output_effort === "string"
              ? legacyExtraBody.output_effort
              : "max"
        );
        setConversationId(settings.conversationId || generateConversationId());
      } catch {
        window.localStorage.removeItem(STORAGE_KEY);
        setConversationId(generateConversationId());
      }
    } else {
      setConversationId(generateConversationId());
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) {
      return;
    }

    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        apiKey: rememberKey ? apiKey : "",
        rememberKey,
        modelPreset,
        model: currentModelId,
        systemPrompt,
        temperature,
        maxTokens,
        minBalanceGuard,
        maxAttachmentMb,
        webSearch,
        reasoningEffort,
        verbosity,
        outputEffort,
        conversationId,
      })
    );
  }, [
    apiKey,
    rememberKey,
    modelPreset,
    currentModelId,
    systemPrompt,
    temperature,
    maxTokens,
    minBalanceGuard,
    maxAttachmentMb,
    webSearch,
    reasoningEffort,
    verbosity,
    outputEffort,
    conversationId,
    hydrated,
  ]);

  useEffect(() => {
    if (hydrated && conversationId) {
      fetchState().catch(() => {});
    }
  }, [hydrated, conversationId]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }

    const timer = window.setInterval(async () => {
      if (!apiKey.trim()) {
        return;
      }
      try {
        await refreshAll({ syncRemote: true });
      } catch {
        // Quiet background sync.
      }
    }, AUTO_SYNC_INTERVAL_MS);

    return () => window.clearInterval(timer);
  }, [apiKey, conversationId, hydrated]);

  useEffect(() => {
    if (activeFolderId === "__all__" || activeFolderId === "__unfiled__") {
      return;
    }
    if (!folders.some((folder) => folder.id === activeFolderId)) {
      setActiveFolderId("__all__");
    }
  }, [activeFolderId, folders]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }

    const limit = currentMaxOutputTokens;
    if (!maxTokens) {
      return;
    }

    const numericMaxTokens = Number(maxTokens);
    if (Number.isFinite(numericMaxTokens) && numericMaxTokens > limit) {
      setMaxTokens(String(limit));
    }
  }, [modelPreset, currentMaxOutputTokens, maxTokens, hydrated]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }

    if (currentModelId === "gpt-5.4-pro") {
      if (!["medium", "high", "xhigh"].includes(reasoningEffort)) {
        setReasoningEffort("xhigh");
      }
      if (!["low", "medium", "high"].includes(verbosity)) {
        setVerbosity("high");
      }
      return;
    }

    if (currentModelId === "claude-opus-4.6" && !["none", "low", "medium", "high", "max"].includes(outputEffort)) {
      setOutputEffort("max");
    }
  }, [currentModelId, hydrated, outputEffort, reasoningEffort, verbosity]);

  function renderConversationLibraryItem(item) {
    const isActive = item.id === conversationId;

    return (
      <button
        key={item.id}
        type="button"
        draggable
        className={`mac-list-item${isActive ? " active" : ""}`}
        onDragStart={(event) => {
          setDraggedConversationId(item.id);
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setData("text/plain", item.id);
          event.currentTarget.style.opacity = "0.52";
        }}
        onDragEnd={(event) => {
          event.currentTarget.style.opacity = "1";
          setDraggedConversationId(null);
          setDragOverFolderId(null);
        }}
        onClick={() => loadConversation(item)}
      >
        <div className="mac-list-title">{item.title || "新对话"}</div>
        <div className="mac-list-meta">{item.updatedAtLabel || ""}</div>
        <div className="mac-list-excerpt">{item.latestExcerpt || (item.isEmpty ? "还没有消息" : "无摘要")}</div>
      </button>
    );
  }

  function renderMessage(message, index) {
    const role = message.role === "user" ? "user" : message.role === "system" ? "system" : "assistant";
    const structured = extractStructuredContent(message.content);
    const reasoningText = typeof message.reasoningContent === "string" ? message.reasoningContent.trim() : "";
    const hasRecoveredReasoning = role === "assistant" && Boolean(reasoningText);
    const openReasoning = hasRecoveredReasoning && !structured.text;

    return (
      <div key={message.messageId || `${role}-${index}`} className={`message-row message-${role}`}>
        <div className="bubble">
          {structured.text ? <div className="bubble-text">{structured.text}</div> : null}
          {structured.attachments.length ? (
            <div className="bubble-attachments">
              {structured.attachments.map((attachment) => renderAttachmentPreview(attachment))}
            </div>
          ) : null}
          {hasRecoveredReasoning ? (
            <details className="bubble-reasoning" open={openReasoning}>
              <summary>{structured.text ? "查看已保存的推理草稿" : "查看已恢复的推理草稿"}</summary>
              <pre>{reasoningText}</pre>
            </details>
          ) : null}
        </div>
      </div>
    );
  }

  if (!hydrated) {
    return null;
  }

  return (
    <div className="app-layout">
      <aside className={`sidebar sidebar-left${sidebarOpen ? " is-open" : ""}`}>
        <div className="sidebar-mobile-bar">
          <span>Poe Desk</span>
          <button type="button" className="btn-icon" onClick={() => setSidebarOpen(false)} aria-label="关闭侧栏">
            <X size={16} />
          </button>
        </div>

        <div className="mac-sidebar-header">
          <div className="account-mini-card" title={`最近同步: ${latestBalance?.capturedAtLabel || "未同步"}`}>
            <div className="account-icon">
              <Wallet size={16} />
            </div>
            <div className="account-info">
              <span className="account-title">Poe 积分余额</span>
              <span className="account-balance">{latestBalance?.currentPointBalance ?? "---"}</span>
            </div>
            <button type="button" className="btn-icon-subtle" onClick={handleManualSync} aria-label="同步余额">
              <RefreshCw size={14} />
            </button>
          </div>
        </div>

        <div className="scroll-view sidebar-scroll">
          <div className="sidebar-toolbar">
            <div className="mac-field folder-filter">
              <select value={activeFolderId} onChange={(event) => setActiveFolderId(event.target.value)} className="folder-select">
                <option value="__all__">所有会话</option>
                <option value="__unfiled__">未归档</option>
                {folders.length > 0 ? (
                  <optgroup label="我的文件夹">
                    {folders.map((folder) => (
                      <option key={folder.id} value={folder.id}>
                        {folder.name}
                      </option>
                    ))}
                  </optgroup>
                ) : null}
              </select>
            </div>

            <div className="toolbar-actions">
              <button
                type="button"
                className="btn-icon-subtle"
                onClick={handleCreateFolderPrompt}
                disabled={folderBusy}
                aria-label="新建文件夹"
                title="新建文件夹"
              >
                <FolderPlus size={16} />
              </button>
              <button
                type="button"
                className="btn-icon-subtle"
                onClick={startNewConversation}
                aria-label="新建会话"
                title="新建会话"
              >
                <SquarePen size={16} />
              </button>
            </div>
          </div>

          <section className="nav-section nav-section-conversations">
            {activeFolder ? (
              <div className="active-folder-meta">
                <span className="folder-meta-text">当前选中: {activeFolder.name}</span>
                <button type="button" className="text-danger" onClick={() => deleteFolder(activeFolder)}>
                  删除此夹
                </button>
              </div>
            ) : null}

            <div
              className={`conversation-library${dragOverFolderId === activeFolderId && activeFolderId !== "__all__" ? " is-drop-target" : ""}`}
              onDragOver={(event) => {
                if (activeFolderId === "__all__") {
                  return;
                }
                event.preventDefault();
                event.dataTransfer.dropEffect = "move";
                setDragOverFolderId(activeFolderId);
              }}
              onDragLeave={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget)) {
                  setDragOverFolderId((current) => (current === activeFolderId ? null : current));
                }
              }}
              onDrop={(event) => {
                event.preventDefault();
                const targetConversationId = draggedConversationId || event.dataTransfer.getData("text/plain");
                if (activeFolderId === "__all__") {
                  setDragOverFolderId(null);
                  setStatusText("请先选择目标文件夹");
                  return;
                }
                moveConversationToFolder(targetConversationId, activeFolderId);
              }}
            >
              {visibleConversations.length === 0 ? (
                <div className="empty-text conversation-empty">列表为空</div>
              ) : (
                visibleConversations.map(renderConversationLibraryItem)
              )}
            </div>
          </section>
        </div>
      </aside>

      <button
        type="button"
        className={`mobile-sidebar-backdrop${sidebarOpen || settingsOpen ? " is-visible" : ""}`}
        onClick={() => {
          setSidebarOpen(false);
          setSettingsOpen(false);
        }}
        aria-label="关闭面板"
      />

      <main className="chat-area">
        <div className="mac-header">
          <div className="header-left">
            <button
              type="button"
              className="btn-icon mobile-sidebar-toggle"
              onClick={() => {
                setSidebarOpen(true);
                setSettingsOpen(false);
              }}
              aria-label="打开侧栏"
            >
              <PanelLeft size={18} />
            </button>

            <div className="header-titles">
              <h2>{currentModelLabel}</h2>
              <span className="header-subtitle">
                {statusText}
                {currentConversationCost > 0 ? ` · 消耗 ${currentConversationCost} 积分` : ""}
              </span>
            </div>
          </div>

          <div className="header-actions">
            <button
              type="button"
              className={`btn-icon${settingsOpen ? " active" : ""}`}
              onClick={() => {
                setSettingsOpen((current) => !current);
                setSidebarOpen(false);
              }}
              aria-label="打开设置"
            >
              <Settings size={18} />
            </button>
          </div>
        </div>

        <div className="scroll-view chat-scroll" ref={chatLogRef}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">
                <MessageSquare size={48} strokeWidth={1.4} />
              </div>
              <h3>开始对话</h3>
              <p>有什么我可以帮你的？</p>
            </div>
          ) : (
            messages.map((message, index) => renderMessage(message, index))
          )}
        </div>

        <div className="mac-composer-area">
          {pendingAttachments.length > 0 ? (
            <div className="attachment-tray">
              {pendingAttachments.map((attachment) => (
                <div key={attachment.id} className="attachment-pill">
                  {attachment.kind === "image" ? (
                    <img src={attachment.dataUrl} alt={attachment.name} />
                  ) : (
                    <Paperclip size={14} />
                  )}
                  <span>{attachment.name}</span>
                  <button
                    type="button"
                    onClick={() => setPendingAttachments((current) => current.filter((item) => item.id !== attachment.id))}
                    aria-label={`移除 ${attachment.name}`}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          ) : null}

          <form className="mac-composer-box" onSubmit={handleSubmit}>
            <button
              type="button"
              className="mac-btn-attach"
              onClick={() => fileInputRef.current?.click()}
              disabled={busy}
              aria-label="添加附件"
            >
              <Plus size={18} />
            </button>

            <input
              ref={fileInputRef}
              name="attachments"
              type="file"
              multiple
              hidden
              onChange={async (event) => {
                try {
                  await handleAddAttachments(event.target.files);
                } catch (error) {
                  setStatusText(error.message);
                }
              }}
            />

            <textarea
              ref={userInputRef}
              className="mac-composer-input"
              name="userInput"
              rows={1}
              value={userInput}
              onChange={(event) => {
                setUserInput(event.target.value);
                event.target.style.height = "auto";
                event.target.style.height = `${Math.min(event.target.scrollHeight, 200)}px`;
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="发送消息"
              disabled={busy}
            />

            {busy ? (
              <button type="button" className="mac-btn-stop" onClick={stopCurrentRequest} aria-label="停止">
                <Square size={14} fill="currentColor" />
              </button>
            ) : (
              <button
                type="submit"
                className="mac-btn-send"
                disabled={!userInput.trim() && !pendingAttachments.length}
                aria-label="发送"
              >
                <ArrowUp size={18} />
              </button>
            )}
          </form>
        </div>
      </main>

      <aside className={`sidebar sidebar-right${settingsOpen ? " is-open" : " is-closed"}`}>
        <div className="mac-header">
          <h2>设置与日志</h2>
          <button type="button" className="btn-icon" onClick={() => setSettingsOpen(false)} aria-label="关闭设置">
            <X size={18} />
          </button>
        </div>

        <div className="scroll-view sidebar-scroll inspector-scroll">
          <section className="settings-section">
            <h3 className="section-title">模型配置</h3>

            <div className="settings-card">
              <div className="settings-card-row">
                <div className="mac-field">
                  <label htmlFor="api-key-input">API Key</label>
                  <input
                    id="api-key-input"
                    type="password"
                    value={apiKey}
                    onChange={(event) => setApiKey(event.target.value)}
                    placeholder="pk-..."
                    autoComplete="off"
                  />
                </div>
              </div>

              <label className="mac-switch-row settings-card-row" htmlFor="remember-key">
                <span>记住密钥</span>
                <div className="mac-switch">
                  <input
                    id="remember-key"
                    type="checkbox"
                    checked={rememberKey}
                    onChange={(event) => setRememberKey(event.target.checked)}
                  />
                  <span className="slider"></span>
                </div>
              </label>

              <div className="settings-card-row">
                <div className="mac-field">
                  <label htmlFor="model-preset-select">预设模型</label>
                  <select id="model-preset-select" value={modelPreset} onChange={(event) => setModelPreset(event.target.value)}>
                    <option value="gpt54pro">GPT-5.4-Pro</option>
                    <option value="claudeopus46">Claude Opus 4.6</option>
                    <option value="geminiflashlite">Gemini 2.0 Flash Lite</option>
                  </select>
                </div>
              </div>

              <div className="settings-card-row">
                <div className="mac-field">
                  <label htmlFor="system-prompt-input">系统提示词</label>
                  <textarea
                    id="system-prompt-input"
                    rows={3}
                    value={systemPrompt}
                    onChange={(event) => setSystemPrompt(event.target.value)}
                    placeholder="你是一个得力的助手。"
                  />
                </div>
              </div>
            </div>

            <div className="settings-card settings-card-grid">
              <div className="settings-grid">
                <div className="mac-field">
                  <label htmlFor="temperature-input">温度</label>
                  <input
                    id="temperature-input"
                    type="number"
                    min="0"
                    max="2"
                    step="0.1"
                    value={temperature}
                    onChange={(event) => setTemperature(event.target.value)}
                    placeholder="默认"
                  />
                </div>
                <div className="mac-field">
                  <label htmlFor="max-tokens-input">输出上限</label>
                  <input
                    id="max-tokens-input"
                    type="number"
                    min="1"
                    max={currentMaxOutputTokens}
                    step="1"
                    value={maxTokens}
                    onChange={(event) => setMaxTokens(event.target.value)}
                    placeholder={String(currentMaxOutputTokens)}
                  />
                  <span className="mac-field-hint">该模型最大支持 {currentMaxOutputTokens}</span>
                </div>
              </div>
            </div>

            <div className="settings-card settings-card-grid">
              <div className="settings-grid">
                <div className="mac-field">
                  <label htmlFor="balance-guard-input">最低余额</label>
                  <input
                    id="balance-guard-input"
                    type="number"
                    min="0"
                    step="1"
                    value={minBalanceGuard}
                    onChange={(event) => setMinBalanceGuard(event.target.value)}
                    placeholder="可选"
                  />
                </div>
                <div className="mac-field">
                  <label htmlFor="attachment-limit-input">附件上限 MB</label>
                  <input
                    id="attachment-limit-input"
                    type="number"
                    min="1"
                    step="1"
                    value={maxAttachmentMb}
                    onChange={(event) => setMaxAttachmentMb(event.target.value)}
                  />
                </div>
              </div>
            </div>

            <div className="settings-card">
              {currentModelId === "gpt-5.4-pro" ? (
                <>
                  <label className="mac-switch-row settings-card-row" htmlFor="web-search-toggle">
                    <span>联网搜索</span>
                    <div className="mac-switch">
                      <input
                        id="web-search-toggle"
                        type="checkbox"
                        checked={webSearch}
                        onChange={(event) => setWebSearch(event.target.checked)}
                      />
                      <span className="slider"></span>
                    </div>
                  </label>

                  <div className="settings-card-row">
                    <div className="mac-field">
                      <label htmlFor="reasoning-effort-select">推理力度</label>
                      <select
                        id="reasoning-effort-select"
                        value={reasoningEffort}
                        onChange={(event) => setReasoningEffort(event.target.value)}
                      >
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                        <option value="xhigh">XHigh</option>
                      </select>
                    </div>
                  </div>

                  <div className="settings-card-row">
                    <div className="mac-field">
                      <label htmlFor="verbosity-select">详细程度</label>
                      <select id="verbosity-select" value={verbosity} onChange={(event) => setVerbosity(event.target.value)}>
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                      </select>
                      <span className="mac-field-hint">将按 Poe 当前允许的最高档位发送模型专属参数。</span>
                    </div>
                  </div>
                </>
              ) : null}

              {currentModelId === "claude-opus-4.6" ? (
                <>
                  <label className="mac-switch-row settings-card-row" htmlFor="web-search-toggle">
                    <span>联网搜索</span>
                    <div className="mac-switch">
                      <input
                        id="web-search-toggle"
                        type="checkbox"
                        checked={webSearch}
                        onChange={(event) => setWebSearch(event.target.checked)}
                      />
                      <span className="slider"></span>
                    </div>
                  </label>

                  <div className="settings-card-row">
                    <div className="mac-field">
                      <label htmlFor="output-effort-select">输出力度</label>
                      <select
                        id="output-effort-select"
                        value={outputEffort}
                        onChange={(event) => setOutputEffort(event.target.value)}
                      >
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                        <option value="none">None</option>
                        <option value="max">Max</option>
                      </select>
                      <span className="mac-field-hint">Claude 当前支持联网搜索与输出力度两个高级参数。</span>
                    </div>
                  </div>
                </>
              ) : null}

              {currentModelId === "gemini-2.0-flash-lite" ? (
                <div className="settings-card-row">
                  <span className="mac-field-hint danger-text model-inline-hint">该模型暂不支持高级参数。</span>
                </div>
              ) : null}
            </div>
          </section>

          <section className="settings-section">
            <h3 className="section-title">对账与审计</h3>

            <div className="audit-stats-grid">
              <div className="stat-box">
                <span className="stat-val success">{auditSummary?.reconcileCounts?.matched || 0}</span>
                <span className="stat-label">已对上</span>
              </div>
              <div className="stat-box">
                <span className="stat-val warning">{auditSummary?.reconcileCounts?.pending || 0}</span>
                <span className="stat-label">待核对</span>
              </div>
              <div className="stat-box">
                <span className="stat-val danger">{auditSummary?.reconcileCounts?.unmatched || 0}</span>
                <span className="stat-label">异常差异</span>
              </div>
            </div>

            <div className="request-log-list">
              {recentRequests.length === 0 ? (
                <div className="empty-text">暂无历史</div>
              ) : (
                recentRequests.map((request) => (
                  <div key={request.id} className="log-card">
                    <div className="log-header">
                      <span className="log-model">{request.model}</span>
                      <div className="log-badges">
                        <span
                          className={`log-badge ${
                            request.status === "completed" ? "green" : request.status === "failed" ? "red" : "gray"
                          }`}
                        >
                          {humanizeStatus(request.status)}
                        </span>
                        <span
                          className={`log-badge ${
                            request.reconcileStatus === "matched"
                              ? "green"
                              : request.reconcileStatus === "unmatched"
                                ? "red"
                                : "gray"
                          }`}
                        >
                          {humanizeReconcileStatus(request.reconcileStatus)}
                        </span>
                      </div>
                    </div>

                    <div className="log-meta">
                      {request.createdAtLabel || "未知时间"}
                      {request.usageCostPoints !== null && request.usageCostPoints !== undefined ? (
                        <strong> · 消耗 {request.usageCostPoints}</strong>
                      ) : null}
                    </div>

                    <div className="log-excerpt">{request.excerpt || "无文本摘要"}</div>

                    <div className="log-actions">
                      <button type="button" onClick={() => exportEvidence(request.id)}>
                        <FolderArchive size={12} /> 证据包
                      </button>
                      <button type="button" className="text-danger" onClick={() => deleteRequest(request.id)}>
                        <Trash2 size={12} /> 删除
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </aside>
    </div>
  );
}
