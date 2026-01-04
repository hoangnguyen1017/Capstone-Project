import nodemailer from "nodemailer";

export default async function sendMail(
    to: string,
    subject: string,
    text: string,
    attachments?: any[]
) {
    const transporter = nodemailer.createTransport({
        service: "gmail",
        auth: {
            user: process.env.GMAIL_USER!,
            pass: process.env.GMAIL_PASS!,
        },
    });

    await transporter.sendMail({
        from: process.env.GMAIL_USER!,
        to,
        subject,
        text,
        attachments, 
    });
}
