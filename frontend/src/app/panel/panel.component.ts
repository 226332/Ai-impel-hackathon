import {Component, ElementRef, OnInit, ViewChild} from '@angular/core';
import {NgxFileDropEntry, FileSystemFileEntry, FileSystemDirectoryEntry} from 'ngx-file-drop';
import {StateService} from '../state.service';
import {animate, style, transition, trigger} from '@angular/animations';
import {HttpClient} from '@angular/common/http';
import {BACKEND_URL} from '../constants';
import {environment} from '../../environments/environment';

@Component({
  selector: 'app-panel',
  templateUrl: './panel.component.html',
  styleUrls: ['./panel.component.css'],
  animations: [
    trigger('slideInOut', [
      transition(':enter', [
        style({transform: 'translateY(-300%)'}),
        animate('2000ms ease-out', style({transform: 'translateY(0%)'}))
      ]),
      transition(':leave', [
        animate('500ms ease-in', style({transform: 'translateY(-500%)'}))
      ])
    ]),
    trigger('slideInOut2', [
      transition(':enter', [
        style({transform: 'translateY(500%)'}),
        animate('2000ms ease-out', style({transform: 'translateY(0%)'}))
      ]),
      transition(':leave', [
        animate('500ms ease-in', style({transform: 'translateY(500%)'}))
      ])
    ])
  ]
})
export class PanelComponent implements OnInit {
  get d() {
    return this.files.length > 0 || this.filesFire.length>0;
  }
  path:string;
  env
  @ViewChild('myCanvas', {static: true}) myCanvas: any;

  constructor(public state: StateService, private http:HttpClient) {
    this.env = environment.production;
    state.getD = () => {
      return this.d;
    };
  }

  ngOnInit() {

  }

  public files: NgxFileDropEntry[] = [];
  public filesFire: NgxFileDropEntry[] = [];

  public fileOver(files: NgxFileDropEntry[]) {
    this.state.overNormal=true;
  }

  reset(){
    this.state.setToNormalState()
    this.state.clicked=false;
    this.files=[]
    this.filesFire=[]
    this.state.resultReady=false;

  }

  public fileLeave(files: NgxFileDropEntry[]) {
    this.state.overNormal=false;
  }
  public fileOverFire(files: NgxFileDropEntry[]) {
    this.state.overFire=true;
  }

  public fileLeaveFire(files: NgxFileDropEntry[]) {
    this.state.overFire=false;
  }

  public dropped(files: NgxFileDropEntry[]) {
    this.files = files;
    this.state.setToGreenState()
    for (const droppedFile of files) {

      // Is it a file?
      if (droppedFile.fileEntry.isFile) {
        const fileEntry = droppedFile.fileEntry as FileSystemFileEntry;
        fileEntry.file((file: File) => {

          // Here you can access the real file
          console.log(droppedFile.relativePath, file);

          /**
           // You could upload it like this:
           const formData = new FormData()
           formData.append('logo', file, relativePath)

           // Headers
           const headers = new HttpHeaders({
            'security-token': 'mytoken'
          })

           this.http.post('https://mybackend.com/api/upload/sanitize-and-save-logo', formData, { headers: headers, responseType: 'blob' })
           .subscribe(data => {
            // Sanitized logo returned from backend
          })
           **/

        });
      } else {
        // It was a directory (empty directories are added, otherwise only files)
        const fileEntry = droppedFile.fileEntry as FileSystemDirectoryEntry;
        console.log(droppedFile.relativePath, fileEntry);
      }
    }
  }
  promiseFile = async (file): Promise<File> => {
    return new Promise(((resolve, reject) => {
      file.file(d => {
        resolve(d);
      });
    }));
  };
  public async droppedFire(files: NgxFileDropEntry[]) {
    this.filesFire = files;
    this.state.setToFireState();
    for (const droppedFile of files) {
    }



  }

  async onBtnClick() {
    // if (this.state.isGreenMode){this.state.setToFireState()}
    // if (this.state.isFireMode){this.state.setToGreenState()}
    const formData = new FormData()
    let df = undefined;
    let f = undefined;
    if (this.state.isGreenMode){
      df = this.files[0]

    }

    if (this.state.isFireMode){
      df = this.filesFire[0]

    }
    f = await this.promiseFile(df.fileEntry as FileSystemFileEntry);
    formData.append('file',f , 'anyName')
    setTimeout(() => {
      this.state.clicked = true;
    }, 500);
    let result:any = await this.http.post(`http://localhost:5000/inference?type=${this.state.isFireMode?'fire':'green'}`, formData).toPromise();
    console.log(result)
    this.path = result.filename;
    setTimeout ( ()=>{
      this.state.resultReady=true;
    }, 500)
  }



}
