(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        green - block
        purple - block
        orange - block
        red - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (on green yellow)
        (on purple green)
        (on orange purple)
        (on red orange)
        (on blue red)
        (clear blue)
        ; yellow -> green -> purple -> orange -> red -> blue
    )
    (:goal (and (on orange purple)
        (on yellow orange)
        (on green yellow)
        (on blue green)
        (on red blue)))
)
    