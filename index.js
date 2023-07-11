// 1. 必要なモジュールとライブラリをインポートする
import { OpenAI } from 'langchain/llms';
import { RetrievalQAChain } from 'langchain/chains';
import { HNSWLib } from 'langchain/vectorstores';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from 'fs';
import * as dotenv from 'dotenv';
import readline from 'readline';

// 2. 環境変数を読み込む
dotenv.config();

// 3. 入力データとパスを設定する
const txtFilename = "LangChain";
const txtPath = `./${txtFilename}.txt`;
const VECTOR_STORE_PATH = `${txtFilename}.index`;

// 4. メインの関数 runWithEmbeddings を定義する
export const runWithEmbeddings = async () => {
  // 5. 空の設定オブジェクトで OpenAI モデルを初期化する
  const model = new OpenAI({});

  // 6. ベクトルストアファイルが存在するかどうかをチェックする
  let vectorStore;
  if (fs.existsSync(VECTOR_STORE_PATH)) {
    // 6.1. ベクトルストアファイルが存在する場合は、メモリに読み込む
    console.log('ベクトルストアが存在します..');
    vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
  } else {
    // 6.2. ベクトルストアファイルが存在しない場合は、作成する
    // 6.2.1. 入力テキストファイルを読み込む
    const text = fs.readFileSync(txtPath, 'utf8');
    // 6.2.2. 指定されたチャンクサイズで RecursiveCharacterTextSplitter を作成する
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
    // 6.2.3. 入力テキストをドキュメントに分割する
    const docs = await textSplitter.createDocuments([text]);
    // 6.2.4. OpenAIEmbeddings を使用してドキュメントから新しいベクトルストアを作成する
    vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
    // 6.2.5. ベクトルストアをファイルに保存する
    await vectorStore.save(VECTOR_STORE_PATH);
  }

  // 7. 初期化された OpenAI モデルとベクトルストアリトリーバーを渡して RetrievalQAChain を作成する
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

  // 8. ユーザーに質問を入力してもらう
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  rl.question('質問を入力してください:', async (userQuestion) => {
    // 9. 入力の質問で RetrievalQAChain を呼び出し、結果を 'res' 変数に格納する
    const res = await chain.call({
      query: userQuestion,
    });

    // 10. 結果をコンソールに出力する
    console.log('結果:', res);

    // 11. readline インターフェースを閉じる
    rl.close();
  });
};

// 12. メインの関数 runWithEmbeddings を実行する
runWithEmbeddings();
