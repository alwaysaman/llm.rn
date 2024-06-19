import React, { useState, useRef, useEffect } from 'react'
import type { ReactNode } from 'react'
import { Platform } from 'react-native'
import { SafeAreaProvider } from 'react-native-safe-area-context'
import DocumentPicker from 'react-native-document-picker'
import type { DocumentPickerResponse } from 'react-native-document-picker'
import { Chat, darkTheme, defaultTheme } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import ReactNativeBlobUtil from 'react-native-blob-util'
// eslint-disable-next-line import/no-unresolved
import { initLlama, LlamaContext, convertJsonSchemaToGrammar } from 'llama.rn'
import { Bubble } from './Bubble'

const { dirs } = ReactNativeBlobUtil.fs

const randId = () => Math.random().toString(36).substr(2, 9)
// console.log(randId());

const user = { id: 'y9d7f8pgn' }

const systemId = 'h3o3lc5xj'
const system = { id: systemId }

const initialChatPrompt =
  'This is a conversation between user and BawabaBot, a friendly chatbot working with EmiratesNBD. respond in simple markdown.\n\n'

const generateChatPrompt = (
  context: LlamaContext | undefined,
  conversationId: string,
  messages: MessageType.Any[],
) => {
  const prompt = [...messages]
    .reverse()
    .map((msg) => {
      if (
        !msg.metadata?.system &&
        msg.metadata?.conversationId === conversationId &&
        msg.metadata?.contextId === context?.id &&
        msg.type === 'text'
      ) {
        return `${msg.author.id === systemId ? '<|assistant|>' : '<|user|>'}${msg.text}<|end|>\n<|assistant|>`
      }
      return ''
    })
    .filter(Boolean)
    .join('\n')
  return initialChatPrompt + prompt
} 

const defaultConversationId = 'default'

const renderBubble = ({
  child,
  message,
}: {
  child: ReactNode
  message: MessageType.Any
}) => <Bubble child={child} message={message} />

export default function App() {
  const [context, setContext] = useState<LlamaContext | undefined>(undefined)

  const [inferencing, setInferencing] = useState<boolean>(false)
  const [messages, setMessages] = useState<MessageType.Any[]>([])

  const conversationIdRef = useRef<string>(defaultConversationId)

  useEffect(() => {
    const loadModel = async () => {
      console.log('loading model');
      initLlama({
        model: '/Users/aman_pocs/Desktop/llm_inference/phi3_app/assets/model1.gguf',
        use_mlock: true,
        n_gpu_layers: Platform.OS === 'ios' ? 0 : 0, // > 0: enable GPU
        // embedding: true,
      })
      // .catch(err => console.log(err))
      .then((ctx) => {
        setContext(ctx)
        addSystemMessage(
          `Hello There !!! How can I help you today?` 
        )
      })
      .catch((err) => {
        addSystemMessage(`Context initialization failed: ${err.message}`)
      })
      
    }
    loadModel();
  }, []);

  const addMessage = (message: MessageType.Any, batching = false) => {
    if (batching) {
      // This can avoid the message duplication in a same batch
      setMessages([message, ...messages])
    } else {
      setMessages((msgs) => [message, ...msgs])
    }
  }

  const addSystemMessage = (text: string, metadata = {} ) => {
    const textMessage: MessageType.Text = {
      author: system,
      createdAt: Date.now(),
      id: randId(),
      text,
      type: 'text',
      metadata: { system: true, ...metadata },
    }
    addMessage(textMessage)
  }
  const handleSendPress = async (message: MessageType.PartialText) => {
    const textMessage: MessageType.Text = {
      author: user,
      createdAt: Date.now(),
      id: randId(),
      text: message.text,
      type: 'text',
      metadata: {
        contextId: context?.id,
        conversationId: conversationIdRef.current,
      },
    }
    addMessage(textMessage)
    setInferencing(true)

    const id = randId()
    const createdAt = Date.now()
    let prompt = '<|user|>\n'
    prompt += generateChatPrompt(context, conversationIdRef.current, [
      textMessage,
      ...messages,
    ])    

    let grammar
    {
      // Test JSON Schema -> grammar
      const schema = {
        oneOf: [
          {
            type: 'object',
            properties: {
              function: { const: 'create_event' },
              arguments: {
                type: 'object',
                properties: {
                  title: { type: 'string' },
                  date: { type: 'string' },
                  time: { type: 'string' },
                },
              },
            },
          },
          {
            type: 'object',
            properties: {
              function: { const: 'image_search' },
              arguments: {
                type: 'object',
                properties: {
                  query: { type: 'string' },
                },
              },
            },
          },
        ],
      }

      const converted = convertJsonSchemaToGrammar({
        schema,
        propOrder: { function: 0, arguments: 1 },
      })
      // @ts-ignore
      if (false) console.log('Converted grammar:', converted)
      grammar = undefined
      // Uncomment to test:
      // grammar = converted
    }
    context
      ?.completion(
        {
          prompt,
          // n_predict: 400,
          // temperature: 0.7,
          // top_k: 40, // <= 0 to use vocab size
          // top_p: 0.5, // 1.0 = disabled
          // tfs_z: 1.0, // 1.0 = disabled
          // typical_p: 1.0, // 1.0 = disabled
          // penalty_last_n: 256, // 0 = disable penalty, -1 = context size
          // penalty_repeat: 1.18, // 1.0 = disabled
          // penalty_freq: 0.0, // 0.0 = disabled
          // penalty_present: 0.0, // 0.0 = disabled
          // mirostat: 0, // 0/1/2
          // mirostat_tau: 5, // target entropy
          // mirostat_eta: 0.1, // learning rate
          // penalize_nl: false, // penalize newlines
          // seed: 1234, // random seed
          // n_probs: 0, // Show probabilities
          stop: ['<|end|>','<|user|>'],
          // grammar,
          // n_threads: 4,
          // logit_bias: [[15043,1.0]],
        },
        (data) => {
          const { token } = data
          setMessages((msgs) => {
            const index = msgs.findIndex((msg) => msg.id === id)
            if (index >= 0) {
              return msgs.map((msg, i) => {
                if (msg.type == 'text' && i === index) {
                  return {
                    ...msg,
                    text: (msg.text + token).replace(/^\s+/, ''),
                  }
                }
                return msg
              })
            }
            return [
              {
                author: system,
                createdAt,
                id,
                text: token,
                type: 'text',
                metadata: { contextId: context?.id },
              },
              ...msgs,
            ]
          })
        },
      )
      .then((completionResult) => {
        console.log('completionResult: ', completionResult)
        const timings = `${completionResult.timings.predicted_per_token_ms.toFixed()}ms per token, ${completionResult.timings.predicted_per_second.toFixed(
          2,
        )} tokens per second`
        setMessages((msgs) => {
          const index = msgs.findIndex((msg) => msg.id === id)
          if (index >= 0) {
            return msgs.map((msg, i) => {
              if (msg.type == 'text' && i === index) {
                return {
                  ...msg,
                  metadata: {
                    ...msg.metadata,
                    timings,
                  },
                }
              }
              return msg
            })
          }
          return msgs
        })
        setInferencing(false)
      })
      .catch((e) => {
        console.log('completion error: ', e)
        setInferencing(false)
        addSystemMessage(`Completion failed: ${e.message}`)
      })
  }

  return (
    <SafeAreaProvider>
      <Chat
        renderBubble={renderBubble}
        theme={defaultTheme}
        messages={messages}
        onSendPress={handleSendPress}
        user={user}
        textInputProps={{
          editable: !!context,
          placeholder: !context
            ? 'Press the file icon to pick a model'
            : 'Type your message here',
        }}
      />
    </SafeAreaProvider>
  )
}