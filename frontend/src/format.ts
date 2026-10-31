// Presentation only. Values used by charts, validation and APIs retain precision.
export function decimal(value:number):string {
 return Number(value.toFixed(2)).toFixed(2);
}
export function metric(value:number):string {
 return Number.isInteger(value)?value.toLocaleString('en-US'):Number(value.toFixed(2)).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
}
export function decimalText(text:string):string {
 // Preserve quoted source categories and identifiers; round numeric prose only.
 return text.replace(/"(?:[^"\\]|\\.)*"|(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+(?:e[+-]?\d+)?(?![\w.])/gi,token=>{
  if(token.startsWith('"'))return token;
  const number=Number(token.replaceAll(',','')),rounded=Number(number.toFixed(2));
  return (token.startsWith('+')&&rounded>0?'+':'')+decimal(number);
 });
}
