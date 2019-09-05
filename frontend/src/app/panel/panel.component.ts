import {Component, ElementRef, OnInit, ViewChild} from '@angular/core';
import {NgxFileDropEntry, FileSystemFileEntry, FileSystemDirectoryEntry} from 'ngx-file-drop';
import {StateService} from '../state.service';
import {animate, style, transition, trigger} from '@angular/animations';
import {HttpClient} from '@angular/common/http';
import {BACKEND_URL} from '../constants';

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

  @ViewChild('myCanvas', {static: true}) myCanvas: any;

  constructor(private state: StateService, private http:HttpClient) {
    state.getD = () => {
      return this.d;
    };
  }

  ngOnInit() {
    this.dupa();
  }

  public files: NgxFileDropEntry[] = [];
  public filesFire: NgxFileDropEntry[] = [];

  public fileOver(files: NgxFileDropEntry[]) {
    this.state.overNormal=true;
  }

  reset(){
    this.state.setToNormalState()
    this.state.clicked=false;
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
    let result = await this.http.post(`${BACKEND_URL}/xd?type=${this.state.isFireMode?'fire':'green'}`, formData).toPromise();
    console.log(result)
    setTimeout ( ()=>{
      this.state.resultReady=true;
    }, 500)
  }

  dupa() {
    window.requestAnimFrame = (function() {
      return window.requestAnimationFrame ||
        window.webkitRequestAnimationFrame ||
        window.mozRequestAnimationFrame ||
        window.oRequestAnimationFrame ||
        window.msRequestAnimationFrame ||
        function(callback) {
          window.setTimeout(callback, 1000 / 60);
        };
    })();

    Math.randMinMax = function(min, max, round) {
      var val = min + (Math.random() * (max - min));

      if (round) {
        val = Math.round(val);
      }

      return val;
    };
    Math.TO_RAD = Math.PI / 180;
    Math.getAngle = function(x1, y1, x2, y2) {

      var dx = x1 - x2,
        dy = y1 - y2;

      return Math.atan2(dy, dx);
    };
    Math.getDistance = function(x1, y1, x2, y2) {

      var xs = x2 - x1,
        ys = y2 - y1;

      xs *= xs;
      ys *= ys;

      return Math.sqrt(xs + ys);
    };

    var FX = {};
    let _this = this;
    (function() {

      var lastUpdate = new Date(),
        ctx = (<HTMLCanvasElement> _this.myCanvas.nativeElement).getContext('2d'),
        mouseUpdate = new Date(),
        lastMouse = [],
        width, height;

      FX.particles = [];

      setFullscreen();
      document.getElementById('button').addEventListener('mousedown', buttonEffect);

      function buttonEffect() {
        var button = document.getElementById('button');
        let b = button.getBoundingClientRect();
        console.log(b);
        var
          height = button.offsetHeight,
          left = b.left,
          top = b.top,
          width = button.offsetWidth,
          x, y, degree;

        for (var i = 0; i < 40; i = i + 1) {

          if (Math.random() < 0.5) {

            y = Math.randMinMax(top, top + height);

            if (Math.random() < 0.5) {
              x = left;
              degree = Math.randMinMax(-45, 45);
            } else {
              x = left + width;
              degree = Math.randMinMax(135, 225);
            }

          } else {

            x = Math.randMinMax(left, left + width);

            if (Math.random() < 0.5) {
              y = top;
              degree = Math.randMinMax(45, 135);
            } else {
              y = top + height;
              degree = Math.randMinMax(-135, -45);
            }

          }
          createParticle({
            x: x,
            y: y,
            degree: degree,
            speed: Math.randMinMax(100, 150),
            vs: Math.randMinMax(-4, -1)
          });
        }
      }

      window.setTimeout(buttonEffect, 100);

      loop();

      window.addEventListener('resize', setFullscreen);

      function createParticle(args) {

        var options = {
          x: width / 2,
          y: height / 2,
          color: `rgba(50,50,50,${Math.randMinMax(1, 15) / 100.0})`,
          degree: Math.randMinMax(0, 360),
          speed: Math.randMinMax(300, 350),
          vd: Math.randMinMax(-90, 90),
          vs: Math.randMinMax(-8, -5)
        };

        for (const key in args) {
          options[key] = args[key];
        }

        FX.particles.push(options);
      }

      function loop() {

        var thisUpdate = new Date(),
          delta = (lastUpdate - thisUpdate) / 1000,
          amount = FX.particles.length,
          size = 2,
          i = 0,
          p;

        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.fillRect(0, 0, width, height);
        // ctx.globalCompositeOperation = "screen";
        ctx.globalCompositeStyle = 'lighter';

        for (; i < amount; i = i + 1) {

          p = FX.particles[i];

          p.degree += (p.vd * delta);
          p.speed += (p.vs);// * delta);
          if (p.speed < 0) {
            continue;
          }

          p.x += Math.cos(p.degree * Math.TO_RAD) * (p.speed * delta);
          p.y += Math.sin(p.degree * Math.TO_RAD) * (p.speed * delta);

          ctx.save();

          ctx.translate(p.x, p.y);
          ctx.rotate(p.degree * Math.TO_RAD);

          ctx.fillStyle = p.color;
          ctx.fillRect(-size, -size, size * 2, size * 2);

          ctx.restore();
        }

        lastUpdate = thisUpdate;

        requestAnimFrame(loop);
      }

      function setFullscreen() {
        width = _this.myCanvas.nativeElement.width = window.innerWidth;
        height = _this.myCanvas.nativeElement.height = window.innerHeight;
      };
    })();


  }

}
